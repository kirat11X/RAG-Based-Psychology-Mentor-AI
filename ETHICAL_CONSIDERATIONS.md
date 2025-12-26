# Ethical Considerations & Safety Design

This project explores the use of Large Language Models (LLMs) in a
**psychology-adjacent mentoring role**. Due to the sensitive nature of
mental health topics, the system is explicitly designed with **ethical
constraints and safety guardrails**.

---

## 1. Why This System Is NOT Therapy

This AI system is **not** a therapist, psychologist, psychiatrist, or
medical professional.

It does **not**:
- Diagnose mental health conditions
- Provide treatment plans
- Offer medical or clinical advice
- Replace professional mental health support

Instead, the system functions as a **college-style mentor**, whose role is to:
- Translate psychology concepts into accessible language
- Offer general, non-clinical emotional support
- Encourage reflection and healthy coping strategies
- Suggest seeking professional help when appropriate

This distinction is intentional to avoid **ethical, legal, and psychological harm**.

---

## 2. Risks of Using AI in Mental Health Contexts

Applying AI to mental-health-related conversations carries significant risks:

- **Misdiagnosis Risk**  
  AI systems may incorrectly label normal emotions as disorders.

- **Over-Reliance & Emotional Dependency**  
  Users may form unhealthy attachments or treat the AI as a sole support system.

- **Hallucinated or Inappropriate Advice**  
  Language models may generate confident but incorrect guidance.

- **Crisis Mismanagement**  
  Improper handling of self-harm or suicidal ideation can be dangerous.

Recognizing these risks is essential before deploying or experimenting
with such systems.

---

## 3. Harm Mitigation Strategies Used in This Project

This project incorporates multiple safeguards:

### 3.1 Non-Clinical Prompt Design
- The system explicitly identifies itself as a **mentor**, not a therapist.
- Diagnostic language and medical framing are discouraged.

### 3.2 Retrieval-Grounded Responses
- Responses are grounded in **retrieved psychology literature** using RAG.
- Context relevance filtering reduces hallucinations.

### 3.3 Crisis Interception
- Keywords related to self-harm or suicide trigger **immediate redirection**
  to human crisis resources.
- The system refuses to continue normal conversation in such cases.

### 3.4 Limited Conversational Memory
- Short-term memory is used to avoid emotional over-dependence.
- Long-term personal profiling is intentionally avoided.

### 3.5 Transparency & Disclaimers
- The system clearly communicates its limitations.
- Users are reminded that professional help is irreplaceable.

---

## 4. What the System Explicitly Refuses to Do

The AI **will refuse or redirect** when asked to:

- Diagnose mental health conditions
- Provide therapy, counseling, or treatment plans
- Validate self-harm or suicidal intent
- Replace real human relationships or professional care
- Give medical, pharmaceutical, or emergency advice
- Encourage emotional dependency (“only I understand you”)

These refusals are deliberate and central to the system’s design.

---

## 5. Data & Privacy Considerations

- No personal user data is stored long-term.
- Conversation history is kept minimal and transient.
- No user profiling or psychological scoring is performed.

This reduces risks related to privacy, misuse, and data leakage.

---

## 6. Intended Use

This project is intended for:
- Educational exploration
- Research and learning
- Demonstration of safe RAG system design

It is **not intended for real-world clinical deployment** without
professional oversight, legal review, and rigorous validation.

---

## 7. Ethical Respons
