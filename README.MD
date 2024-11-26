# Set up environment

**Pre-requisites :**

1.  Create virtual environment (I'm using [venv](https://docs.python.org/3/library/venv.html))
2.  Run `pip install -r requirements.txt`
3.  Download and install [Ollama](https://ollama.com/)

**Ollama**
After installing, open Ollama and run command `ollama run llama3.2` on your terminal to download the pretrained LLM model. In this project, I'm using `llama3.2-3B` because it's lightweight and has good performance.

## Deployment

I'm using Streamlit and Docker to deploy the chatbot.

### Step 1 : Build Docker Image

```bash
  docker build -t ecommerce-chatbot-app .
```

### Step 2 : Run Docker Container

```bash
  docker run -d --restart always --gpus all --name ecommerce-chatbot-app -p 8501:8501 ecommerce-chatbot-app
```

To stop and delete the container run

```bash
  docker stop ecommerce-chatbot-app && docker rm ecommerce-chatbot-app
```

## Chatbot Overview

**Example of Questions**
![alt text](https://github.com/AFauzulh/simple-ecommerce-chatbot/blob/main/assets/question_1.png?raw=true)

![alt text](https://github.com/AFauzulh/simple-ecommerce-chatbot/blob/main/assets/question_2.png?raw=true)

**Retrieved data from API**

![alt text](https://github.com/AFauzulh/simple-ecommerce-chatbot/blob/main/assets/retrieved_information.png?raw=true)
