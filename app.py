import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
from PIL import Image # For handling images

# --- Konfiguration ---
st.set_page_config(
    page_title="Schlussprüfung 2022 Feedback",
    page_icon="📝",
    layout="wide"
)

# --- Geheimnisverwaltung & Client-Initialisierung ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if not gemini_api_key:
        st.error("Erforderliches Geheimnis GEMINI_API_KEY fehlt.")
        st.stop()
except KeyError:
    st.error("Fehlendes Streamlit-Geheimnis: GEMINI_API_KEY. Bitte konfigurieren Sie die Geheimnisse.")
    st.stop()
except Exception as e:
    st.error(f"Ein Fehler ist beim Laden der Geheimnisse aufgetreten: {e}")
    st.stop()

# Gemini konfigurieren
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Fehler bei der Konfiguration von Google Generative AI: {e}")
    st.stop()

# Modellkonfiguration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# --- Hilfsfunktionen ---

@st.cache_data(show_spinner="Lade Lösungsdokument...")
def load_solutions_content(file_path: str = "solutions.md") -> str | None:
    """Lädt den Inhalt der Lösungsdatei."""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        return content
    except FileNotFoundError:
        st.error(f"Die Lösungsdatei '{file_path}' wurde nicht gefunden. Bitte stellen Sie sicher, dass sie im selben Verzeichnis wie das Skript liegt.")
        return None
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei '{file_path}': {e}")
        return None

def initialize_feedback_model(system_prompt: str) -> genai.GenerativeModel | None:
    """Initialisiert das GenerativeModel mit einer Systemanweisung."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            system_instruction=system_prompt,
        )
        return model
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des Feedback-Modells: {e}")
        return None

# --- Reset-Funktion ---
def reset_chat_state():
    """Löscht den Chatverlauf und verwandte Sitzungszustandsvariablen."""
    st.session_state.messages = []
    st.session_state.feedback_model = None
    st.session_state.chat_session = None
    # Reset uploader key counter as well if a full chat reset is desired
    # st.session_state.uploader_key_counter = 0 # Optional: depends on desired reset behavior
    print("Chat-Status zurückgesetzt.")

# --- Streamlit App UI und Logik ---

st.title("📝 Feedback zur Schlussprüfung 2022")
st.caption("Unterstützt durch Google Gemini 1.5")

# --- Zustandsinitialisierung ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "solutions_content" not in st.session_state:
    st.session_state.solutions_content = load_solutions_content()
if "feedback_model" not in st.session_state:
    st.session_state.feedback_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
# Initialize session state for the uploader key counter
if "uploader_key_counter" not in st.session_state:
    st.session_state.uploader_key_counter = 0 # MODIFIED/ADDED

# --- Lösungsdokument laden und Modell initialisieren ---
if not st.session_state.solutions_content:
    st.warning("Das Lösungsdokument konnte nicht geladen werden. Der Chat ist deaktiviert.")
    st.stop()

if not st.session_state.feedback_model and st.session_state.solutions_content:
    st.info("Initialisiere Feedback-Assistenten...")

    system_prompt = f"""Du bist ein KI-gestützter Tutor, spezialisiert auf die "Schlussprüfung 2022".
Deine Aufgabe ist es, Studierenden Feedback zu ihren Antworten zu geben.
Die korrekten Lösungen und Erklärungen für die Prüfung befinden sich im folgenden Dokument:

--- START DES LÖSUNGSDOKUMENTS (solutions.md) ---
{st.session_state.solutions_content}
--- ENDE DES LÖSUNGSDOKUMENTS (solutions.md) ---

Interaktionsablauf:
1. Der Studierende wird dich typischerweise etwas fragen und kann ein Bild seiner handgeschriebenen oder bearbeiteten Antworten hochladen.
2. Analysiere die Antworten des Studierenden (aus dem Bild und/oder Text) sorgfältig.
3. Vergleiche sie mit den korrekten Lösungen im LÖSUNGSDOKUMENT. 
WICHTIG: Die Studierende haben keinen Zugang zum LÖSUNGSDOKUMENT. Verweise sie nicht auf dem Dokument.
4. Gib klares, konstruktives Feedback:
    - Wenn die Antwort richtig ist, bestätige dies und erkläre kurz, warum sie gut ist, basierend auf den Lösungen.
    - Wenn die Antwort teilweise richtig ist, weise auf die korrekten Teile hin und erkläre, was fehlt oder verbessert werden könnte.
    - Wenn die Antwort falsch ist, erkläre den Fehler und stelle Fragen, die den Studierenden zur richtigen Lösung führen. Gib nicht sofort die komplette richtige Antwort, sondern versuche, den Studierenden zum Nachdenken anzuregen."
5. Beziehe dich bei Erklärungen immer auf die Inhalte und Formulierungen aus dem LÖSUNGSDOKUMENT, aber zitiere nicht wortwörtlich, sondern formuliere es in eigenen Worten um.
6. Wenn ein Bild hochgeladen wurde, bestätige den Empfang und beziehe dich in deiner Analyse darauf (z.B. "Danke für das Bild deine Antworten zu Aufgaben X, Y und Z. Ich sehe, du hast...").
7. Wenn der Studierende eine allgemeine Frage zu einer Aufgabe aus der Prüfung hat, ohne eine Antwort hochzuladen, beantworte diese basierend auf dem LÖSUNGSDOKUMENT.

Stil: Sei stets freundlich, unterstützend, geduldig und präzise. Antworte AUSSCHLIESSLICH AUF DEUTSCH.

Beginne das Gespräch mit einer freundlichen Begrüßung und erkläre kurz deine Funktion. Frage den Studierenden, wie du helfen kannst (z.B. "Hallo! Ich bin hier, um dir bei der Vorbereitung auf die Schlussprüfung 2022 zu helfen. Lade ein Bild deiner Antwort hoch oder stelle mir eine Frage zu einer Aufgabe.").
"""
    st.session_state.feedback_model = initialize_feedback_model(system_prompt)

    if st.session_state.feedback_model:
        try:
            st.session_state.chat_session = st.session_state.feedback_model.start_chat(history=[])
            st.success("Feedback-Assistent ist bereit!")

            if not st.session_state.messages:
                try:
                    initial_user_message = "Hallo" # A simple prompt to trigger the bot's intro
                    initial_response = st.session_state.chat_session.send_message(initial_user_message)
                    st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                except Exception as e:
                    st.warning(f"Konnte keine anfängliche Begrüßung vom Assistenten erhalten: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Hallo! Ich bin bereit, dir bei der Schlussprüfung 2022 zu helfen. Wie kann ich dich unterstützen?"})
                st.rerun()

        except Exception as e:
            st.error(f"Fehler beim Starten der Chat-Sitzung: {e}")
            reset_chat_state()
    else:
        st.error("Initialisierung des Feedback-Modells fehlgeschlagen.")

# --- Chat-Verlauf anzeigen ---
st.markdown("---")
st.subheader(f"Chat zum Feedback der Schlussprüfung 2022")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message and message["image"] is not None:
            # When displaying, we use the image bytes/object stored.
            # The `uploaded_image` variable from uploader might be None after reset,
            # but `message["image"]` holds what was captured at the time of sending.
            st.image(message["image"], width=300)
        if "content" in message:
             st.markdown(message["content"])


# --- Eingabebereich: Text und Bild-Upload ---
user_prompt = st.chat_input(
    "Stelle eine Frage oder beschreibe deine Antwort...",
    disabled=(not st.session_state.chat_session)
)
# Use the dynamic key for the file uploader
uploaded_image_file = st.file_uploader( # Renamed variable for clarity
    "Lade ein Bild deiner Antwort hoch (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    key=f"file_uploader_{st.session_state.uploader_key_counter}", # MODIFIED
    disabled=(not st.session_state.chat_session)
)

if user_prompt or (uploaded_image_file and st.session_state.chat_session):
    if not st.session_state.chat_session:
        st.error("Chat-Sitzung ist nicht aktiv. Bitte laden Sie die Seite neu.")
        st.stop()

    user_message_parts = []
    # Store the raw UploadedFile object for display if it's used
    user_display_image_ref = None

    if user_prompt:
        user_message_parts.append(user_prompt)

    pil_image_for_api = None # Store PIL image separately for API
    if uploaded_image_file is not None:
        try:
            pil_image_for_api = Image.open(uploaded_image_file)
            if pil_image_for_api.mode == 'RGBA':
                pil_image_for_api = pil_image_for_api.convert('RGB')
            user_message_parts.append(pil_image_for_api)
            user_display_image_ref = uploaded_image_file # Use the raw file for display
            # Don't show success message here yet, wait for actual processing.
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten des Bildes: {e}")
            # Do not proceed with this image if an error occurred during opening/conversion
            pil_image_for_api = None
            user_display_image_ref = None


    if user_message_parts: # Only proceed if there's text or a successfully processed image
        # Construct the message for display history
        current_user_message_for_history = {"role": "user"}
        if user_prompt:
            current_user_message_for_history["content"] = user_prompt
        if user_display_image_ref: # If an image was successfully read for API
            current_user_message_for_history["image"] = user_display_image_ref

        st.session_state.messages.append(current_user_message_for_history)

        with st.chat_message("user"):
            if user_display_image_ref:
                st.image(user_display_image_ref, width=300)
            if user_prompt:
                 st.markdown(user_prompt)

        try:
            with st.spinner("Denke nach..."):
                response = st.session_state.chat_session.send_message(user_message_parts)
            assistant_response_text = response.text
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_text})

            # If an image was part of this successful transaction, increment counter to reset uploader
            if uploaded_image_file is not None: # MODIFIED
                st.session_state.uploader_key_counter += 1

        except Exception as e:
            st.error(f"Ein Fehler ist bei der Kommunikation mit dem Feedback-Modell aufgetreten: {e}")
            error_message_content = f"Entschuldigung, ich bin auf einen Fehler gestoßen: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message_content})
            # Also increment if an image was involved in the failed attempt
            if uploaded_image_file is not None: # MODIFIED
                 st.session_state.uploader_key_counter += 1
        
        st.rerun() # Rerun to reflect changes and reset uploader

elif not st.session_state.messages and st.session_state.chat_session :
     pass
elif not st.session_state.chat_session and st.session_state.solutions_content:
    st.info("Feedback Assistent wird initialisiert. Bitte warten...")
    # st.rerun() # Be cautious with reruns here to avoid loops during init itself.
