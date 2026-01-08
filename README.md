# Intelligent_resume_enhancer
Enhances resume content to match job description and makes it ATS friendly using LLM, OpenSearch

# AI-Powered Intelligent Resume Builder (AWS)

An **event-driven, serverless AI system** that intelligently analyzes resumes, performs semantic similarity using embeddings, and generates **job-optimized, ATS-friendly resumes** using large language models on AWS.

This project demonstrates how to build a **production-grade LLM application** using **Amazon Bedrock, OpenSearch vector search, and serverless AWS services**.

---

## 🚀 Key Features

* Secure resume upload and storage
* Semantic understanding using embeddings
* Vector similarity search for relevant resume sections
* AI-powered resume generation using LLMs
* Event-driven, decoupled architecture
* Asynchronous user notifications
* Scalable and cost-efficient serverless design

---

## 🧠 High-Level Architecture

<img width="1024" height="1024" alt="Architecture_resume_generator" src="https://github.com/user-attachments/assets/c0116747-89af-43d1-ab6c-2ced87ca7001" />


**Flow Overview:**

1. User uploads resume via UI
2. Resume is stored in S3 and metadata captured
3. Embeddings are generated and stored in OpenSearch
4. Relevant content is retrieved using vector similarity
5. LLM generates an optimized resume
6. User is notified when processing is complete

**Core Architecture Pattern:**

* Event-driven (EventBridge)
* Serverless compute (AWS Lambda)
* AI/ML powered (Amazon Bedrock)

---

## 🏗️ AWS Services Used

| Category        | Service                                    |
| --------------- | ------------------------------------------ |
| Frontend        | Streamlit / FastAPI                        |
| Authentication  | Amazon Cognito                             |
| API Layer       | Amazon API Gateway                         |
| Orchestration   | Amazon EventBridge                         |
| Compute         | AWS Lambda                                 |
| AI / LLM        | Amazon Bedrock (Claude Sonnet, Embeddings) |
| Vector Database | Amazon OpenSearch                          |
| Storage         | Amazon S3                                  |
| Metadata Store  | Amazon DynamoDB                            |
| Notifications   | Amazon SNS                                 |
| Observability   | CloudWatch, X-Ray                          |

---

## 🔄 Event-Driven Workflow

### 1️⃣ Resume Upload

* Resume uploaded via UI
* Stored in S3
* Metadata saved in DynamoDB
* `ResumeUploaded` event published to EventBridge
* SNS notification sent for upload status

### 2️⃣ Embedding Creation

* EventBridge triggers embedding Lambda
* Resume text extracted
* Embeddings generated using Bedrock
* Stored in OpenSearch vector index
* `EmbeddingsCreated` event published

### 3️⃣ Resume Generation

* Vector similarity search in OpenSearch
* Relevant content passed to Claude Sonnet
* AI-generated resume created
* Output stored in S3
* Stats updated in DynamoDB
* SNS notification sent to user

---

## 📣 Notifications (SNS)

SNS is used to notify users and systems asynchronously:

* Resume upload success/failure
* Embedding generation status
* Resume generation completion
* Error and operational alerts

This enables seamless integration with:

* Email
* Slack
* Webhooks
* UI status updates

---

## 🔐 Security & Best Practices

* IAM least-privilege roles
* Encryption at rest (S3, DynamoDB, OpenSearch)
* Cognito-based authentication
* Event-driven decoupling
* Retry and failure handling
* Scalable and cost-efficient design

---

## 🧩 Why This Architecture?

* **Scalable**: Fully serverless and async
* **Extensible**: Easy to add new AI agents or steps
* **Production-ready**: Observability, security, notifications
* **Modern AI stack**: LLMs + embeddings + vector search

---

## 🔮 Future Enhancements

* Bedrock Agents for agentic workflows
* Job description analyzer agent
* Resume match scoring (0–100)
* Multi-language resume support
* ATS scoring and feedback loop
* Resume versioning and comparison

---

## 📌 Use Cases

* Job seekers optimizing resumes
* Recruiters shortlisting candidates
* AI-powered HR platforms
* Resume screening and personalization tools

---

## 🛠️ Tech Stack

`AWS Lambda` · `Amazon Bedrock` · `Claude Sonnet` · `OpenSearch` · `EventBridge` · `SNS` · `DynamoDB` · `S3` · `FastAPI` · `Streamlit`

---

## 🤝 Contributing

Contributions, ideas, and discussions are welcome!
Feel free to open issues or submit pull requests.

Linkedin- www.linkedin.com/in/subhra-ojha-3685a14b

⭐ If you find this project useful, consider starring the repository!

