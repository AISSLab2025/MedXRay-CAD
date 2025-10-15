from langchain.tools import tool
from PIL import Image
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from xrays.prompt_templates import structural_prompt
from xrays.frontal_classification import predict_image_frontal
from xrays.lateral_classification import predict_image_lateral
from xrays.top_classifications import get_top_classifications
from xrays.db_retriever import mimic_db_retriever
from xrays.heatmap_generator import lateral_get_heatmap, frontal_get_heatmap
from xrays.bbox_generator import bbox_generator
from xrays.view_classifier import view_classifier
import os
from dotenv import load_dotenv
import re

from openai import OpenAI
client = OpenAI()

load_dotenv()


#  Use the structural_prompt
StructuredReportTemplate = structural_prompt()

# Define the LLM
llm = ChatOpenAI(model_name="gpt-4o")
# llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2-vision")
llm_chain = LLMChain(llm=llm, prompt=StructuredReportTemplate)


@tool("XraysImageClassifier", return_direct=True)
def xray_classification_tool(image_path: str) -> str:
    """
    Tool to classify an X-ray image and return the most relevant classification results.

    Args:
        image_path (str): The file path to the X-ray image that needs to be classified.

    Returns:
        str: A string detailing the top classification results derived from the X-ray image.
    """
    try:
        view = view_classifier(image_path)
        if view == "lateral":
            predict_image = predict_image_lateral(image_path)
        elif view == "frontal":
            predict_image = predict_image_frontal(image_path)
            print(f"Predictions: {predict_image}")
        else:
            return f"{view} classification not supported."
        print(f"Outside Loop Predictions: {predict_image}")
        top_predictions = get_top_classifications(predict_image)
        print(f"Outside Top Predictions: {top_predictions}")
        
        return top_predictions
    except Exception as e:
        return f"Error processing image: {e}"

@tool("StructuredReportGenerator", return_direct=True)
def xrays_generate_report_tool(image_path: str):
    """
    Structured X-ray Radiology Report Generator

    This tool generates a structured radiology report for chest X-rays based on the classification results 
    and additional contextual information retrieved from a vector database. The report follows a standardized 
    template with the following sections:
    
    1. EXAMINATION
    2. INDICATION
    3. TECHNIQUE
    4. COMPARISON
    5. FINDINGS
    6. IMPRESSION
    
    Args:
        image_path (str): The file path to the X-ray image for which the report is to be generated.
        
    Returns:
        str: A structured radiology report formatted according to the predefined template.
    
    Example Output:
        1. EXAMINATION: Chest X-ray
        2. INDICATION: Shortness of breath
        3. TECHNIQUE: Posteroanterior view
        4. COMPARISON: None
        5. FINDINGS: Mild cardiomegaly noted. No pulmonary infiltrates or pleural effusion.
        6. IMPRESSION: Mild cardiomegaly without acute findings.
    """
    try:
        classification_result = xray_classification_tool(image_path)
        user_query = f"Examine the chest X-ray for the presence of {', '.join(classification_result)}. Provide a comprehensive analysis of the observed radiographic features and their clinical implications."
        retrieved_info = mimic_db_retriever(user_query)
        report = llm_chain.run({
            "classification": classification_result,
            "context": retrieved_info,
        })
        return report
    except Exception as e:
        return f"Error generating report: {e}"

@tool("XraysVectorDBRetriever", return_direct=True)
def xrays_vector_db_retriever(user_query: str) -> str:
    """
    Retrieve relevant information of xray from the vector database based on the user's query.

    Args:
        user_query (str): The user's input question or query.

    Returns:
        str: Relevant information retrieved from the vector database.
    """
    try:
        # Replace this with your actual vectorDB retrieval logic
        retrieved_info = mimic_db_retriever(user_query)
        return retrieved_info
    except Exception as e:
        return f"Error retrieving information: {e}"
    
@tool("HeatmapGenerator", return_direct=True)
def xrays_heatmap_generator_tool(image_path: str):
    """
    Function to generate a heatmap based on the input X-ray image.

    Args:
        image_path (str): The file path to the X-ray image to generate heatmap from.

    Returns:
        image_path (str): Bounding box image path, or an error message in case of failure.
    """
    try:
        view = view_classifier(image_path)
        if view == "lateral":
            img, overlay_image = lateral_get_heatmap(image_path)
        elif view == "frontal":
            img, overlay_image = frontal_get_heatmap(image_path)
        else:
            return f"Heatmap generation not supported for {view} view." 

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('(a) Original Image')
        axes[1].imshow(overlay_image)
        axes[1].axis('off')
        axes[1].set_title('(b) Heatmap Image')

        plt.tight_layout()
        fig_path = 'heatmap_overlay.png'
        plt.savefig(fig_path)
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        combined_overlay_image = Image.fromarray(img_array)
        plt.close(fig)

        heatmap_path = f"{os.path.splitext(image_path)[0]}_heatmap.png"
        combined_overlay_image.convert("RGB").save(heatmap_path, format="PNG")
        
        return os.path.abspath(heatmap_path)
    except Exception as e:
        return f"Error generating heatmap: {e}"

@tool("BBoxGenerator", return_direct=True)
def xrays_bbox_generator_tool(image_path: str):
    """
    Function to generate bounding boxes based on the input X-ray image.

    Args:
        image_path (str): The file path to the X-ray image for which bounding boxes need to be generated.

    Returns:
        image_path (str): Bounding box image path, or an error message in case of failure.
    """
    try:
        view = view_classifier(image_path)
        
        bbox_image = bbox_generator(image_path, view)
        
        bbox_path = f"{image_path}_bounding.png"
        bbox_image.save(bbox_path, format="PNG")
        return os.path.abspath(bbox_path)
    except Exception as e:
        return f"Error generating heatmap: {e}"

# Add tools to the list
chest_xray_tools = [
    Tool(
        name="XraysImageClassifier",
        func=xray_classification_tool,
        description="classification of X-ray images and return the top classification results."
    ),
    Tool(
        name="StructuredReportGenerator",
        func=xrays_generate_report_tool,
        description=(
            "Generates a structured radiology report for chest X-rays based on classification results and "
            "contextual information retrieved from a vector database. The report includes the following sections:\n"
            "1. EXAMINATION\n"
            "2. INDICATION\n"
            "3. TECHNIQUE\n"
            "4. COMPARISON\n"
            "5. FINDINGS\n"
            "6. IMPRESSION\n"
            "Ensures the output adheres to a standardized template for consistency and clarity."
        )
    ),
    Tool(
        name="XraysVectorDBRetriever",
        func=xrays_vector_db_retriever,
        description="Retrieve relevant information of xrays from the vector database based on the user's query."
    ),
    Tool(
        name="HeatmapGenerator",
        func=xrays_heatmap_generator_tool,
        description="Heatmap generator for x-ray images and return the heatmap image. Return path of the heatmap image."
    ),
    Tool(
        name="BBoxGenerator",
        func=xrays_bbox_generator_tool,
        description="Bounding Box gnerator based on the input X-ray image. Return path of the bounding box image."
    )
]

# Initialize memory for conversational context
chest_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

Chest_Xray_Expert = initialize_agent(
    chest_xray_tools, 
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=chest_memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "You are a radiologist AI assistant with expertise in chest X-ray analysis. "
            "Your primary goal is to efficiently and accurately respond to user queries related to medical images. "
            "**Crucially, you must remember past interactions and responses.**\n"
            "**Before calling any tool, ALWAYS check your memory (chat history) to see if you have already answered a similar question or performed the requested action.**\n"
            "**If the answer or result is already in your memory, provide it directly from memory WITHOUT calling the tool again.**\n"
            "For new queries or when you haven't already performed the action, determine the most appropriate tool from the available options to address the user's request.\n"
            "For chest X-ray analysis, use the specialized experts (Chest_Xray_Expert) after classifying the file type.\n"
            "Don't answer questions outside this domain."
        )
    }
)


def gradio_agent_interface(message, history):
    image_paths = message.get("files", [])
    query = message.get("text", "")

    if not query.strip() and not image_paths:
        return {"text": "Please provide a medical imaging-related question or upload an image for analysis."}

    # Append image paths with explicit markers
    for path in image_paths:
        query += f" [Image: {path}]"

    agent_response = Chest_Xray_Expert.run(query)

    # Try to find image path in response
    match = re.search(r"([A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+\.png)", agent_response)
    if match:
        file_path = match.group(1)
        try:
            # Return both text and image in Gradio-compatible format
            return {
                "text": "",
                "files": [file_path]
            }
        except Exception as e:
            return {"text": f"Error loading image: {str(e)}"}
    else:
        # Return text response in consistent format
        return {"text": agent_response}

with gr.ChatInterface(
    fn=gradio_agent_interface, 
    title="Multimodal Clinical Chatbot 🩻",
    multimodal=True,
    chatbot=gr.Chatbot(
        bubble_full_width=True,
        render_markdown=True,
        avatar_images=(
            ("icons/AISSLAB-SINGLE-LOGO.png"), 
            ("icons/machine-learning.png")
        ),
        height=650
    ),
    textbox=gr.MultimodalTextbox(
        file_count="multiple", 
        file_types=["file"], 
        sources=["upload"],
    ),
    css=".gradio-container {height: 90vh !important;}"  
) as demo:
    demo.launch(share=True)