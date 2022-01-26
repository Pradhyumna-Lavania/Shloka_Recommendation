Sanskrit texts are one of the richest sources of wisdom, spirituality, and science of life. But unfortunately, today, due to a hectic schedule and lack of Sanskrit education, it seems nearly impossible to find the specific topic among thousands of Sanskrit verses manually.
This project aims to solve this problem using machine learning concepts to scan thousands of verses to recommend the verse relevant to the user.
Presently, our project spans over 13000+ Sanskrit verses taken from the Bhagavad Gita, Shrimad Bhagavatam, Ishopanishad, and Brahm Samhita.
The project work can be divided into two broad categories. The first part comprises collecting, storing, and optimizing the data, and the second part consists of implementing machine learning concepts.
Web scrapping means collecting data from the web using automated techniques. Many different methods and tools are available for scrapping the web. However, no matter what technique is used, the approach and the objective remain the same, capture web data and present it in a more structured format. We scraped these verses, their translations, word-to-word meaning, and purport from vedabase.io using the Beautiful Soup Library and organized all the data in CSV format using pandas. The purports available on vedabase.io have some authentic recommendations, and they have been used as the gold standard recommendations to test our model. To get those recommendations from the purport, we utilized the regex library.
In the next part, viz., machine learning, we first separated the verses with gold-standard recommendations from the others.
Vector based models are used to give shlokas a unique identity for application in model.
After some refinement, we implemented standard recommendation models like BM-25, if f, Count Vectorizer, and Embeddings on those verses to get the primary set of recommendations.
Precision, recall, F value, and MRR scores of each model were calculated to find the overall score to decide the best performer.

