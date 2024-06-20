# PDF-Answering-AI
#Idea behind the project

The project ‘pdf answering AI’ is based on developing an AI application that takes a pdf file as an input and gives the answers to the questions of the user.

The final prototype is based on context filtering to get the best context for out model i.e. distilbert-base-cased-distilled-squad which is trained on SQUAD datatset. 

The context filtering is achieved by arranging the pdf text in form of a hierarchical structure where Headings are at the top, then Sub-headings and finally the paragraphs. By creating this hierarchy, it became very easy to find a good context for the question by first finding the best match for the question vector from the heading vectors. After finding a closely matching heading, we go and check for a closely related subheading to the question. Then the closest paragraph to the question becomes the context. This helped reducing the time complexity of the searching of answer as it reduced the context to a mere fraction of the complete corpus. 

