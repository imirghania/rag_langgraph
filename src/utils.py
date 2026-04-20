import re 

def trim_think_with_regex(text):
    """
    Removes content between <think> and </think> tags (inclusive).
    Uses re.DOTALL to match across multiple lines.
    """
    cleaned_text = re.sub(r'<think>.*?</think>', 
                        '', text, flags=re.DOTALL)
    return cleaned_text.strip()