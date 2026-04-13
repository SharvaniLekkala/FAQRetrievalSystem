#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKENS 100
#define MAX_STRING 2048

const char *stop_words[] = {
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "do", "how", "i", "can", "my", "in", "from", "with", "it", "are", 
    "on", "be", "get", "when", "where", "why", "who", "will", "would", "should", "could", NULL
};

const char *medical_entities[] = {"medical", "certificate", "hospital", "doctor", "fever", "dengue", "insurance", "paracetamol", "dosage", "specialist", "vaccination", "blood", "illness", "prescription", NULL};
const char *legal_entities[] = {"legal", "police", "fir", "bail", "arrest", "lawyer", "court", "divorce", "property", "tenant", "eviction", "rights", "rti", "trademark", NULL};
const char *tech_entities[] = {"wi-fi", "router", "password", "computer", "laptop", "battery", "android", "iphone", "apple", "cloud", "storage", "virus", "internet", "ram", "vpn", "bluetooth", "phishing", NULL};
const char *finance_entities[] = {"bank", "account", "cibil", "loan", "tax", "income", "mutual", "fund", "upi", "stock", "deposit", "fraud", "gst", "epf", "credit", "card", NULL};
const char *edu_entities[] = {"university", "leave", "attendance", "examination", "scholarship", "hostel", "degree", "transcript", "admission", "cuet", "semester", "student", NULL};

int is_stop_word(const char *word) {
    for (int i = 0; stop_words[i] != NULL; i++) {
        if (strcmp(word, stop_words[i]) == 0) return 1;
    }
    return 0;
}

void to_lowercase(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

void lemmatize(char *word) {
    int len = strlen(word);
    // Synonym mapping: exam -> examination
    if (strcmp(word, "exam") == 0) {
        strcpy(word, "examination");
        return;
    }
    if (len > 4 && strcmp(word + len - 3, "ing") == 0) word[len - 3] = '\0';
    else if (len > 3 && strcmp(word + len - 2, "es") == 0) word[len - 2] = '\0';
    else if (len > 2 && word[len - 1] == 's') word[len - 1] = '\0';
}

void pos_tag(const char *word, char *pos) {
    int len = strlen(word);
    if (len > 4 && strcmp(word + len - 4, "tion") == 0) strcpy(pos, "N");
    else if (len > 4 && strcmp(word + len - 4, "ment") == 0) strcpy(pos, "N");
    else if (len > 2 && strcmp(word + len - 2, "ed") == 0) strcpy(pos, "V");
    else strcpy(pos, "O");
}

void ner_tag(const char *word, char *ner) {
    strcpy(ner, "O");
    for(int i=0; medical_entities[i]; i++) if(strcmp(word, medical_entities[i])==0) { strcpy(ner, "MEDICAL"); return; }
    for(int i=0; legal_entities[i]; i++) if(strcmp(word, legal_entities[i])==0) { strcpy(ner, "LEGAL"); return; }
    for(int i=0; tech_entities[i]; i++) if(strcmp(word, tech_entities[i])==0) { strcpy(ner, "TECH"); return; }
    for(int i=0; finance_entities[i]; i++) if(strcmp(word, finance_entities[i])==0) { strcpy(ner, "FINANCE"); return; }
    for(int i=0; edu_entities[i]; i++) if(strcmp(word, edu_entities[i])==0) { strcpy(ner, "EDUCATION"); return; }
}

int preprocess(const char *text, char tokens[MAX_TOKENS][64], char pos[MAX_TOKENS][8], char ner[MAX_TOKENS][32]) {
    char temp[MAX_STRING];
    strncpy(temp, text, MAX_STRING - 1);
    temp[MAX_STRING - 1] = '\0';
    to_lowercase(temp);
    
    for (int i = 0; temp[i]; i++) {
        if (!isalnum(temp[i]) && temp[i] != '-') temp[i] = ' ';
    }
    
    int count = 0;
    char *token = strtok(temp, " \t\n\r");
    while (token != NULL && count < MAX_TOKENS) {
        if (strlen(token) > 1 && !is_stop_word(token)) {
            char p[64];
            strncpy(p, token, 63);
            p[63] = '\0';
            lemmatize(p);
            if (!is_stop_word(p) && strlen(p) > 1) {
                strcpy(tokens[count], p);
                pos_tag(p, pos[count]);
                ner_tag(p, ner[count]);
                count++;
            }
        }
        token = strtok(NULL, " \t\n\r");
    }
    return count;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: preprocessor.exe \"your sentence here\"\n");
        return 1;
    }

    char query[MAX_STRING] = "";
    for(int i=1; i<argc; i++) {
        strcat(query, argv[i]);
        if(i < argc-1) strcat(query, " ");
    }

    char tokens[MAX_TOKENS][64];
    char pos[MAX_TOKENS][8];
    char ner[MAX_TOKENS][32];
    int num_tokens = preprocess(query, tokens, pos, ner);

    // Output JSON format
    printf("{\n  \"tokens\": [");
    for(int i=0; i<num_tokens; i++) {
        printf("\"%s\"%s", tokens[i], (i == num_tokens - 1) ? "" : ", ");
    }
    printf("],\n  \"pos\": [");
    for(int i=0; i<num_tokens; i++) {
        printf("\"%s\"%s", pos[i], (i == num_tokens - 1) ? "" : ", ");
    }
    printf("],\n  \"ner\": [");
    for(int i=0; i<num_tokens; i++) {
        printf("\"%s\"%s", ner[i], (i == num_tokens - 1) ? "" : ", ");
    }
    printf("]\n}\n");

    return 0;
}
