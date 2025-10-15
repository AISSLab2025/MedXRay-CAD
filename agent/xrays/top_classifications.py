def get_top_classifications(classifications: dict, top_n: int = 3, threshold: float = 0.50) -> dict:
    # Filter classifications that are above the threshold
    filtered_classifications = {k: v for k, v in classifications.items() if v > threshold}
    
    # Sort the filtered classifications by their score in descending order
    sorted_classifications = sorted(filtered_classifications.items(), key=lambda item: item[1], reverse=True)
    
    # Return the top N classifications
    top_classifications = {k: v for k, v in sorted_classifications[:top_n]}
    
    return top_classifications