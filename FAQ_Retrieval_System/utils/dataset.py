import os
import sys

def parse_dataset(filename):
    """Parses the multi-domain FAQ dataset."""
    faqs = []
    current_domain = "Unknown"
    
    if not os.path.exists(filename):
        print(f"Error: dataset file '{filename}' not found.")
        sys.exit(1)
        
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    temp_q = None
    for line in lines:
        line = line.strip()
        if not line or "Multi-Domain FAQ Dataset" in line:
            continue
            
        # Match Qn. (Question)
        if line.startswith('Q') and '.' in line[:5] and any(c.isdigit() for c in line[:5]):
            temp_q = line.split('.', 1)[1].strip()
        # Match A. (Answer)
        elif line.startswith('A.'):
            if temp_q:
                answer = line[2:].strip()
                faqs.append({
                    "question": temp_q,
                    "answer": answer,
                    "domain": current_domain
                })
                temp_q = None
        else:
            # Possible domain header
            if len(line) < 30 and '.' not in line and '?' not in line:
                current_domain = line
                
    return faqs
