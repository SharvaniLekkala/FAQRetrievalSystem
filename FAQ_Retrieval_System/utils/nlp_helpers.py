import re

def robust_sentence_split(text):
    """Splits text into sentences while ignoring dots in URLs/abbreviations."""
    # Split on dot followed by space and Uppercase letter, or end of line.
    sentences = re.split(r'(?<![A-Z])\.\s+(?=[A-Z])|(?<=\w)\.(?=\s*$)', text)
    return [s.strip() for s in sentences if s.strip()]

def calculate_overlap_score(query_items, target_items):
    """Helper for Stage 4: overlapping score calculation (POS/NER)."""
    if not query_items or not target_items:
        return 0.0
    q_set = set(query_items)
    t_set = set(target_items)
    intersection = q_set.intersection(t_set)
    return len(intersection) / max(len(q_set), 1)

def generate_abstract_answer(top_5, faqs):
    """Stage 6: Aggregates and synthesizes an abstract answer from top candidates."""
    found = any(score > 0.05 for _, score in top_5)
    if not found or top_5[0][1] < 0.1:
        return "No relevant answer found."

    # Aggregate by domain
    top_answers_by_domain = {}
    for idx, score in top_5:
        if score < 0.1: continue
        domain = faqs[idx]["domain"]
        if domain not in top_answers_by_domain:
            top_answers_by_domain[domain] = []
        top_answers_by_domain[domain].append(faqs[idx]["answer"])
        
    # Multi-domain synthesis
    segments = []
    for domain, answers in top_answers_by_domain.items():
        main_answer = answers[0]
        sentences = robust_sentence_split(main_answer)
        summary = sentences[0] if sentences else ""
        
        if len(top_answers_by_domain) > 1:
            segments.append(f"Regarding {domain}: {summary}")
        else:
            summary = " ".join(sentences[:2])
            segments.append(summary)
            
    return " ".join(segments)
