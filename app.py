
# 5. Search Personal Information
def search_data(question):
    normalized_question = normalize(question)

    # Remove common words that do not describe personal information
    stop_words = {
        "what", "is", "are", "my", "your", "the",
        "a", "an", "and", "or", "do", "does",
        "i", "you", "me", "tell", "about", "can",
        "please", "when", "who", "how", "much",
        "many", "to", "of", "in", "on"
    }

    question_words = set(normalized_question.split())
    keywords = question_words - stop_words

    matched_answers = []
    matched_questions = []

    # Find all relevant questions in the CSV
    for _, row in data.iterrows():
        stored_question = normalize(row["question"])
        stored_words = set(stored_question.split())
        stored_keywords = stored_words - stop_words

        # Check whether the question contains relevant keywords
        if keywords.intersection(stored_keywords):
            if stored_question not in matched_questions:
                matched_questions.append(stored_question)
                matched_answers.append(row["answer"])

    # Return all matching answers
    if matched_answers:
        return "\n\n".join(matched_answers)

    return None
