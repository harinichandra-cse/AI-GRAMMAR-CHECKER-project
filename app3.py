import difflib
import random
import re

import ollama
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Grammar Checker", page_icon="✍️")
st.title("✍️ AI Grammar Checker")
st.caption("Powered by Llama 3.2 (Ollama)")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["llama3.2:3b", "llama3.2:1b"])
st.sidebar.info("Keep Ollama running in the background.")

# ------------------------------------------------------------------
# Grammar correction (used by Grammar Checker tab AND Chatbot tab)
# ------------------------------------------------------------------
CORRECT_SYSTEM = (
    "You are a grammar correction tool. Fix ALL grammar, spelling, capitalization "
    "and punctuation mistakes in the user's text. Output ONLY the corrected text. "
    "Do not explain. Do not add comments. Do not answer questions. "
    "Keep the same meaning. If the text is already correct, repeat it exactly."
)

EXAMPLES = [
    ("she dont like to going school on sundays", "She doesn't like going to school on Sundays."),
    ("i has a apple", "I have an apple."),
    ("hlo how r u", "Hello, how are you?"),
    ("He go to market yesterday", "He went to the market yesterday."),
    ("I like tea.", "I like tea."),
]


def correct_text(text, model_name):
    """Return the corrected version of the text (only the sentence)."""
    messages = [{"role": "system", "content": CORRECT_SYSTEM}]
    for wrong, right in EXAMPLES:
        messages.append({"role": "user", "content": wrong})
        messages.append({"role": "assistant", "content": right})
    messages.append({"role": "user", "content": text})

    res = ollama.chat(model=model_name, messages=messages, options={"temperature": 0}, keep_alive="30m")
    out = res["message"]["content"].strip()
    out = re.sub(r"^(corrected( text)?|correction)\s*:\s*", "", out, flags=re.I)
    out = out.strip().strip('"').strip()
    return out or text


def get_changes(original, corrected):
    """Compare word by word and list what was changed."""
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        rows.append(
            {
                "Your text": " ".join(a[i1:i2]) or "(missing word)",
                "Corrected": " ".join(b[j1:j2]) or "(remove)",
            }
        )
    return rows


# ------------------------------------------------------------------
# Chatbot helpers
# ------------------------------------------------------------------
CHAT_SYSTEM = (
    "You are a friendly English-speaking friend. Reply in ONE short, simple sentence. "
    "Never ask a question. Never give options or lists."
)


def chat_reply(ctx, model_name):
    res = ollama.chat(
        model=model_name,
        messages=[{"role": "system", "content": CHAT_SYSTEM}] + ctx[-8:],
        options={"temperature": 0.5},
        keep_alive="30m",
    )
    reply = res["message"]["content"].strip()
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    kept = [s for s in sentences if not s.endswith("?")]  # remove questions
    reply = " ".join(kept[:2]).strip()
    return reply or "That sounds nice!"


# ------------------------------------------------------------------
# Practice quiz data
# ------------------------------------------------------------------
QUESTIONS = [
    {"q": "She ____ to school every day.", "options": ["go", "goes", "going", "gone"],
     "answer": "goes", "tip": "With he/she/it in the present simple, add -s: she goes."},
    {"q": "I ____ a new phone yesterday.", "options": ["buy", "buys", "bought", "buying"],
     "answer": "bought", "tip": "'Yesterday' needs the past tense: bought."},
    {"q": "He ____ playing football now.", "options": ["are", "is", "am", "were"],
     "answer": "is", "tip": "Use 'is' with he/she/it in the present continuous."},
    {"q": "They ____ finished their homework.", "options": ["has", "have", "is", "was"],
     "answer": "have", "tip": "Use 'have' with they/we/you/I in the present perfect."},
    {"q": "____ apple a day keeps the doctor away.", "options": ["A", "An", "The", "No article"],
     "answer": "An", "tip": "Use 'an' before a vowel sound: an apple."},
    {"q": "She is good ____ mathematics.", "options": ["in", "at", "on", "with"],
     "answer": "at", "tip": "We say 'good at' a subject."},
    {"q": "The books ____ on the table.", "options": ["is", "are", "was", "be"],
     "answer": "are", "tip": "'Books' is plural, so use 'are'."},
    {"q": "I have lived here ____ 2015.", "options": ["for", "since", "from", "during"],
     "answer": "since", "tip": "Use 'since' with a starting point in time (2015)."},
    {"q": "If it rains, we ____ at home.", "options": ["stay", "stayed", "will stay", "would stayed"],
     "answer": "will stay", "tip": "First conditional: If + present, will + verb."},
    {"q": "____ going to the market.", "options": ["Their", "There", "They're", "Them"],
     "answer": "They're", "tip": "They're = They are."},
    {"q": "He is ____ than his brother.", "options": ["tall", "taller", "tallest", "more tall"],
     "answer": "taller", "tip": "Use the comparative (-er) when comparing two people."},
    {"q": "I look forward to ____ you.", "options": ["see", "seeing", "saw", "seen"],
     "answer": "seeing", "tip": "After 'look forward to', use the -ing form."},
    {"q": "Which sentence is correct?",
     "options": ["She don't like tea.", "She doesn't like tea.", "She not like tea.", "She doesn't likes tea."],
     "answer": "She doesn't like tea.", "tip": "With she, use 'doesn't' + the base verb."},
    {"q": "Which sentence is correct?",
     "options": ["Its a sunny day.", "It's a sunny day.", "Its' a sunny day.", "It is a sunny days."],
     "answer": "It's a sunny day.", "tip": "It's = It is. 'Its' shows ownership."},
    {"q": "She has ____ her lunch already.", "options": ["eat", "ate", "eaten", "eating"],
     "answer": "eaten", "tip": "Present perfect uses has/have + past participle (eaten)."},
    {"q": "I was born ____ 5th May.", "options": ["in", "on", "at", "by"],
     "answer": "on", "tip": "Use 'on' with specific dates."},
    {"q": "The teacher gave us ____ advice.", "options": ["an", "some", "many", "a"],
     "answer": "some", "tip": "'Advice' is uncountable, so use 'some'."},
    {"q": "He ____ to the market yesterday.", "options": ["go", "goes", "went", "gone"],
     "answer": "went", "tip": "Past tense of 'go' is 'went'."},
    {"q": "We ____ dinner when the guests arrived.", "options": ["had", "were having", "have", "are having"],
     "answer": "were having", "tip": "Past continuous shows an action in progress in the past."},
    {"q": "Which sentence is correct?",
     "options": ["Me and him is friends.", "He and I are friends.", "Him and me are friend.", "He and I is friends."],
     "answer": "He and I are friends.", "tip": "Use subject pronouns (He and I) with 'are'."},
]


def start_quiz(n):
    picked = random.sample(QUESTIONS, n)
    prepared = []
    for q in picked:
        opts = q["options"][:]
        random.shuffle(opts)
        prepared.append({**q, "options": opts})
    st.session_state.quiz = {"qs": prepared, "i": 0, "answers": [], "picked": None}


def pick_option(option):
    z = st.session_state.quiz
    if z is not None and z["picked"] is None:
        z["picked"] = option


def next_question():
    z = st.session_state.quiz
    # ignore double clicks / clicks with no answer picked
    if z is None or z["picked"] is None or z["i"] >= len(z["qs"]):
        return
    z["answers"].append(z["picked"])
    z["picked"] = None
    z["i"] += 1


def reset_quiz():
    st.session_state.quiz = None


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Grammar Checker", "💬 AI Chatbot", "🎯 Practice Quiz"])

# ---------------- Tab 1: Grammar Checker ----------------
with tab1:
    st.subheader("Check your text")
    text = st.text_area("Enter your text:", height=150,
                        placeholder="she dont like to going school on sundays")

    if st.button("Check Grammar"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                with st.spinner("Checking..."):
                    fixed = correct_text(text.strip(), model)

                if fixed.strip() == text.strip():
                    st.success("✅ No mistakes found. Your text is correct!")
                else:
                    st.markdown("**✅ Corrected text:**")
                    st.success(fixed)
                    changes = get_changes(text.strip(), fixed)
                    if changes:
                        st.markdown("**✏️ What was changed:**")
                        st.table(pd.DataFrame(changes))
            except Exception as e:
                st.error(f"Ollama not running or model not pulled.\n\n{e}")

# ---------------- Tab 2: Chatbot ----------------
with tab2:
    st.subheader("Chat in English – I will fix your grammar")
    st.caption("Type a sentence. I will show the corrected version and reply shortly.")

    if "chat" not in st.session_state:
        st.session_state.chat = []   # messages shown on screen
        st.session_state.ctx = []    # clean messages sent to the model

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Type your sentence here...")
    if user_text:
        st.session_state.chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            head = None
            try:
                # Step 1: correct the grammar
                with st.spinner("Correcting your sentence..."):
                    fixed = correct_text(user_text, model)
                if fixed.strip() == user_text.strip():
                    head = "✅ **Your sentence is correct!**"
                else:
                    head = f"✏️ **Corrected:** {fixed}"
                st.markdown(head)

                # Step 2: short friendly reply
                with st.spinner("Writing a reply..."):
                    new_ctx = st.session_state.ctx + [{"role": "user", "content": fixed}]
                    reply = chat_reply(new_ctx, model)
                st.session_state.ctx = new_ctx + [{"role": "assistant", "content": reply}]
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": f"{head}\n\n{reply}"})
            except Exception as e:
                err = (
                    f"⚠️ Could not get an answer from **{model}**.\n\n`{e}`\n\n"
                    "Try the **llama3.2:1b** model in the sidebar, or check that Ollama is running."
                )
                st.error(err)
                saved = f"{head}\n\n{err}" if head else err
                st.session_state.chat.append({"role": "assistant", "content": saved})

    if st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.session_state.ctx = []
        st.rerun()

# ---------------- Tab 3: Practice Quiz ----------------
with tab3:
    st.subheader("Practice your grammar")

    if "quiz" not in st.session_state:
        st.session_state.quiz = None
    quiz = st.session_state.quiz

    # Start screen
    if quiz is None:
        n = st.radio("How many questions?", [5, 10], horizontal=True)
        st.button("🚀 Start Practice", on_click=start_quiz, args=(n,))

    # Question screen
    elif quiz["i"] < len(quiz["qs"]):
        total = len(quiz["qs"])
        q = quiz["qs"][quiz["i"]]
        st.caption(f"Question {quiz['i'] + 1} of {total}")
        st.progress(quiz["i"] / total)
        st.markdown(f"### {q['q']}")

        if quiz["picked"] is None:
            for k, opt in enumerate(q["options"]):
                st.button(opt, key=f"opt_{quiz['i']}_{k}", on_click=pick_option,
                          args=(opt,), use_container_width=True)
        else:
            if quiz["picked"] == q["answer"]:
                st.success(f"✅ Correct! {q['answer']}")
            else:
                st.error(f"❌ Wrong. Your answer: {quiz['picked']}\n\nCorrect answer: {q['answer']}")
            st.info(f"💡 {q['tip']}")
            is_last = quiz["i"] == total - 1
            st.button("📊 See Result" if is_last else "Next Question ➡️", on_click=next_question)

    # Result screen
    else:
        qs = quiz["qs"]
        total = len(qs)
        # make sure answers always has exactly one item per question
        answers = (quiz["answers"] + ["(no answer)"] * total)[:total]
        correct = sum(1 for q, a in zip(qs, answers) if a == q["answer"])
        wrong = total - correct

        st.markdown(f"## 🎉 Your score: {correct} / {total}")
        if correct / total >= 0.8:
            st.success("Excellent work!")
        elif correct / total >= 0.5:
            st.info("Good job! Keep practising.")
        else:
            st.warning("Keep practising, you will improve!")

        chart_df = pd.DataFrame({"Correct": [correct], "Wrong": [wrong]}, index=["Answers"])
        st.bar_chart(chart_df, color=["#6FBF73", "#E57373"])

        review = pd.DataFrame(
            {
                "Question": [q["q"] for q in qs],
                "Your answer": answers,
                "Correct answer": [q["answer"] for q in qs],
                "Result": ["✅" if a == q["answer"] else "❌" for q, a in zip(qs, answers)],
            }
        )
        st.markdown("**Review your answers:**")
        st.dataframe(review, hide_index=True, use_container_width=True)

        st.button("🔄 Practice Again", on_click=reset_quiz)