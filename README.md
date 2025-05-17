# Sentiment Analysis Chatbot
## Text Generation and Sentiment Analysis Application

### Overview
- Utilized Google LLM Modal `Gemini-Pro` for text generation, providing superior response compared to smaller models available in Hugging Face.
- Employed Hugging Face Modal `cardiffnlp/twitter-roberta-base-sentiment-latest` for text classification and sentiment prediction.
- Developed a Streamlit web page to display the classification results.
- Deployed the model on a Streamlit website due to the expiration of AWS free tier.

### Scope of Improvement
- Deployment:
  - Consider deploying the application on AWS EC2 VM.
  - Implement Docker to build the container and utilize AWS ECR to store the private Docker image.
  - Integrate GitHub Actions for continuous integration and continuous deployment (CI/CD) pipeline.

- Model Enhancement:
  - Improve text classification model accuracy by fine-tuning on a custom dataset.
  - Utilize advanced prompt templates for the Gemini-Pro model to achieve desired outputs.

### Future Work
- Connect all components to develop an industry-grade application.
- Explore further optimizations and enhancements to the models and deployment process.






## Sentiment Analysis Report

**Date: 21-04-2024**

**Dataset Overview:**

- Total number of texts/documents analyzed: 54

**Sentiment Distribution:**

- Positive Sentiments: 72.22%
- Negative Sentiments: 12.96%
- Neutral Sentiments: 14.81%
 
![output](https://github.com/shashank297/NLP_chatbot_sentiment/assets/67503481/9901e293-f5c5-4adb-825b-dff0c55ecc0a)



**Sentiments  Examples:**

1. **Positive Sentiments:**
   - With resilience and a positive mindset, I embrace setbacks as opportunities for growth and learning.
   - Yes, I thrive on exploring the unfamiliar and broadening my horizons.
   - Friendship is a precious gift that enriches our lives with love, support, and

2. **Negative Sentiments:**
   - Yes, sadness is an emotion I have experienced.
   - Life's burdens weigh heavy upon my weary soul.
   - Disappointment is like a dark cloud, covering the sun of my expectations.

3. **Neutral Sentiments:**
   - Bad news can be upsetting, but it's important to remember that it can also be an opportunity for growth and learning.
   - Forgiveness is not forgetting, but it is choosing to let go of the hurt and anger that binds us.
   - With a mixture of excitement, curiosity, and perhaps a hint of anxiety.

**Key Findings:**
- According to the model's predictions, approximately `72%` of sentiments are positive
- neutral and negative sentiments account for roughly `14.8%` and `13%` respectively.



**Recommendations:**

- Due to the small dataset size and the tendency of the LLM model to generate positive sentiment statements for various questions, it's crucial to enhance result accuracy.
- Implement fine-tuning of a classification model using our dataset to address this challenge effectively.
- Fine-tuning the model can significantly improve accuracy, leading to more reliable outcomes.

**Conclusion:**

In conclusion, the analysis underscores the importance of addressing the limitations posed by the small dataset and the predisposition of the LLM model towards positive sentiment responses. By leveraging fine-tuning techniques with a classification model, there exists a promising avenue to substantially enhance result accuracy. This proactive approach not only mitigates potential biases but also ensures more dependable and precise outcomes in sentiment analysis tasks.

## Steps to Install the Sentiment Analysis Chatbot Locally

1. **Store Google LLM API Key**: 
   - Store your Google LLM API key in the local environment with the name `GOOGLE_API_KEY`.

2. **Create Virtual Environment**: 
   - Create a virtual environment using conda:
     ```bash
     conda create -p venv python=3.9.19
     ```

3. **Activate the Virtual Environment**: 
   - Activate the created environment using conda:
     ```bash
     conda activate ./venv
     ```

4. **Install Required Python Libraries**: 
   - Install the required Python libraries listed in `requirements.txt` using pip:
     ```bash
     pip install -r requirements.txt
     ```

5. **Run the Chatbot Application**: 
   - Execute the following command in the terminal to run the chatbot application:
     ```bash
     streamlit run qachat.py
     ```

Once these steps are completed, you should have the sentiment analysis chatbot up and running locally on your machine.

