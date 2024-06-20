import streamlit as st
import fitz  # PyMuPDF
import spacy
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp=spacy.load('en_core_web_sm', disable=['ner', 'parser'])
# Load QA model
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        if page_num!=1 and page_num!=2 and page_num!=0:
            text += page.get_text()
    return text

def sentence(pdf_text):
    sentence=pdf_text.split('.')
    for i in range(len(sentence)):
        sentence[i]=sentence[i].replace('\n',' ' )
    return sentence

def tokenize(doc):
    tokenized_sentences = []
    for d in doc:
        tokenized_sentence = ""
        for t in nlp(d):
            if not t.is_stop and not t.is_punct and len(t.text) > 2 and not t.is_space:
                tokenized_sentence += t.lemma_.lower() + " "  # Added a space after each token
        if tokenized_sentence!='':
            tokenized_sentences.append(tokenized_sentence.strip())  # Remove trailing space
    return tokenized_sentences


def extract_text_and_font_sizes_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text_font_size_data = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_font_size_data.append((span["text"], span["size"]))
    return text_font_size_data
def categorize_text_by_font_size(text_font_size_data):
    headings_threshold = 16
    subheadings_threshold = 12
    para_threshold = 8
    document_structure = {}
    current_heading = None
    current_subheading = None

    for text, font_size in text_font_size_data:
        if font_size > headings_threshold:
            if current_heading and not document_structure[current_heading]:
                document_structure[current_heading][current_heading] = []
            document_structure[text] = {}
            current_heading = text
            current_subheading = None
        elif font_size > subheadings_threshold:
            if current_heading:
                document_structure[current_heading][text] = []
                current_subheading = text
        elif font_size > para_threshold:
            if current_heading:
                if current_subheading:
                    document_structure[current_heading][current_subheading].append(text)
                else:
                    document_structure[current_heading][current_heading] = [text]
                    current_subheading = current_heading

    if current_heading and not document_structure[current_heading]:
        document_structure[current_heading][current_heading] = []

    return document_structure

def vectorize_text(document_structure):
    all_texts = []
    index_map = []

    for heading, subheadings in document_structure.items():
        all_texts.append(heading)
        index_map.append((heading, None, None))
        for subheading, texts in subheadings.items():
            all_texts.append(subheading)
            index_map.append((heading, subheading, None))
            for text in texts:
                all_texts.append(text)
                index_map.append((heading, subheading, text))
    return index_map


def find_best_match(tfidf_matrix, question_tfidf, index_map):
    similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    best_match = index_map[best_index]
    
    return best_match, best_similarity, similarities

def get_best_paragraph(similarities, index_map, document_structure, threshold=0.2):
    best_heading, best_subheading, _ = index_map[np.argmax(similarities)]
    
    if best_subheading is None:
        all_text = ' '.join([text for subheadings in document_structure[best_heading].values() for text in subheadings])
    else:
        all_text = ' '.join(document_structure[best_heading][best_subheading])
    
    if np.max(similarities) < threshold:
        all_text = ' '.join([text for heading in document_structure for subheadings in document_structure[heading].values() for text in subheadings])
    
    return all_text

def filtered_context(pdf_path, question_tfidf, tfidf_matrix):
    text_font_size_data = extract_text_and_font_sizes_from_pdf(pdf_path)
    document_structure = categorize_text_by_font_size(text_font_size_data)
    
    index_map = vectorize_text(document_structure)
    best_match, best_similarity, similarities = find_best_match(tfidf_matrix, question_tfidf, index_map)
    
    print(f"Best match: {best_match}")
    print(f"Best similarity: {best_similarity}")
    
    best_paragraph = get_best_paragraph(similarities, index_map, document_structure)
    
    return best_paragraph

def answer_question(contexts, question):
    
    model_name = "distilbert/distilbert-base-cased-distilled-squad"
    question_answering_model = pipeline("question-answering", model=model_name)
    qa_input = {
        'question': question,
        'context': contexts
    }
    result = question_answering_model(qa_input)
    
    return result

def top_answer(pdf_path, question):
    tokenized_texts=tokenize(sentence(extract_text_from_pdf(pdf_path)))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
    question_tokenized=tokenize([question])
    question_tfidf=vectorizer.transform(question_tokenized)
    try:
    # Pair each element with its index
        contexts=filtered_context(pdf_path, question_tfidf, tfidf_matrix)
        print(question)
        return answer_question(contexts, question)
    except Exception as e:
        return str(e)

def main():
    st.title('PDF QA Application')
    st.write('Upload a PDF file and ask a question to get an answer.')

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        st.write('PDF file uploaded successfully!')
        st.write('Now enter your question:')
        question = st.text_input('Question')

        if st.button('Ask'):
            if not question:
                st.warning('Please enter a question.')
            else:
                try:
                    # Save the uploaded file temporarily
                    with open('uploaded_file.pdf', 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    # Pass the correct path to answer_question function
                    answer = top_answer('uploaded_file.pdf', question)
                    st.success(f'Answer: {answer}')
                except Exception as e:
                    st.error(f'Error processing PDF or question: {str(e)}')

if __name__ == "__main__":
    main()
