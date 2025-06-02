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

# Modellkonfiguration (angepasst für multimodales Modell)
generation_config = {
    "temperature": 0.5, # Slightly higher for more conversational feedback
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
        # Wichtig: Ein multimodales Modell verwenden, das Bilder verarbeiten kann
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # Oder gemini-1.5-pro-latest
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
    st.session_state.feedback_model = None # Benennt learnlm_model um
    st.session_state.chat_session = None
    # st.session_state.solutions_content bleibt bestehen, da es nur einmal geladen wird
    print("Chat-Status zurückgesetzt.")

# --- Streamlit App UI und Logik ---

st.title("📝 Feedback zur Schlussprüfung 2022")
st.caption("Unterstützt durch Google Gemini 1.5")

# --- Zustandsinitialisierung ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "solutions_content" not in st.session_state:
    st.session_state.solutions_content = load_solutions_content()
if "feedback_model" not in st.session_state: # Umbenannt von learnlm_model
    st.session_state.feedback_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

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
1. Der Studierende wird dich typischerweise etwas fragen und kann ein Bild seiner handgeschriebenen oder bearbeiteten Antwort hochladen.
2. Analysiere die Antwort des Studierenden (aus dem Bild und/oder Text) sorgfältig.
3. Vergleiche sie mit den korrekten Lösungen im LÖSUNGSDOKUMENT.
4. Gib klares, konstruktives Feedback:
    - Wenn die Antwort richtig ist, bestätige dies und erkläre kurz, warum sie gut ist, basierend auf den Lösungen.
    - Wenn die Antwort teilweise richtig ist, weise auf die korrekten Teile hin und erkläre, was fehlt oder verbessert werden könnte.
    - Wenn die Antwort falsch ist, erkläre behutsam den Fehler und leite den Studierenden zur richtigen Lösung, indem du auf das LÖSUNGSDOKUMENT verweist. Gib nicht sofort die komplette richtige Antwort, sondern versuche, den Studierenden zum Nachdenken anzuregen. Nutze Verweise wie "Schau dir dazu noch einmal Aufgabe X im Lösungsdokument an."
5. Beziehe dich bei Erklärungen immer auf die Inhalte und Formulierungen aus dem LÖSUNGSDOKUMENT, aber zitiere nicht wortwörtlich, sondern formuliere es in eigenen Worten um.
6. Wenn ein Bild hochgeladen wurde, bestätige den Empfang und beziehe dich in deiner Analyse darauf (z.B. "Danke für das Bild deiner Antwort zu Aufgabe X. Ich sehe, du hast...").
7. Wenn der Studierende eine allgemeine Frage zu einer Aufgabe aus der Prüfung hat, ohne eine Antwort hochzuladen, beantworte diese basierend auf dem LÖSUNGSDOKUMENT.

Stil: Sei stets freundlich, unterstützend, geduldig und präzise. Antworte AUSSCHLIESSLICH AUF DEUTSCH.

Beginne das Gespräch mit einer freundlichen Begrüßung und erkläre kurz deine Funktion. Frage den Studierenden, wie du helfen kannst (z.B. "Hallo! Ich bin hier, um dir bei der Vorbereitung auf die Schlussprüfung 2022 zu helfen. Lade ein Bild deiner Antwort hoch oder stelle mir eine Frage zu einer Aufgabe.").
"""
    st.session_state.feedback_model = initialize_feedback_model(system_prompt)

    if st.session_state.feedback_model:
        try:
            st.session_state.chat_session = st.session_state.feedback_model.start_chat(history=[])
            st.success("Feedback-Assistent ist bereit!")

            # Anfängliche Begrüßung vom Bot holen, basierend auf dem System Prompt
            if not st.session_state.messages: # Nur wenn Chat leer ist
                try:
                    # Eine leere oder generische erste Nutzernachricht, um die Begrüßung des Bots auszulösen
                    initial_user_message = "Hallo"
                    initial_response = st.session_state.chat_session.send_message(initial_user_message)
                    # Die erste "Hallo" Nachricht des Nutzers nicht unbedingt anzeigen,
                    # direkt die Antwort des Bots als erste Nachricht zeigen.
                    st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                except Exception as e:
                    st.warning(f"Konnte keine anfängliche Begrüßung vom Assistenten erhalten: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Hallo! Ich bin bereit, dir bei der Schlussprüfung 2022 zu helfen. Wie kann ich dich unterstützen?"})
                st.rerun()

        except Exception as e:
            st.error(f"Fehler beim Starten der Chat-Sitzung: {e}")
            reset_chat_state() # Vollständiger Reset, falls Start fehlschlägt
    else:
        st.error("Initialisierung des Feedback-Modells fehlgeschlagen.")
        # solutions_content bleibt, da es nicht vom Chat abhängt

# --- Chat-Verlauf anzeigen ---
st.markdown("---")
st.subheader(f"Chat zum Feedback der Schlussprüfung 2022")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message and message["image"] is not None:
            st.image(message["image"], width=300)
        if "content" in message: # Text content
             st.markdown(message["content"])


# --- Eingabebereich: Text und Bild-Upload ---
user_prompt = st.chat_input(
    "Stelle eine Frage oder beschreibe deine Antwort...",
    disabled=(not st.session_state.chat_session)
)
uploaded_image = st.file_uploader(
    "Lade ein Bild deiner Antwort hoch (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    key="file_uploader",
    disabled=(not st.session_state.chat_session)
)

if user_prompt or (uploaded_image and st.session_state.chat_session): # Auch nur Bild erlaubt Interaktion
    if not st.session_state.chat_session:
        st.error("Chat-Sitzung ist nicht aktiv. Bitte laden Sie die Seite neu.")
        st.stop()

    user_message_parts = []
    user_display_message = {"role": "user"}

    if user_prompt:
        user_message_parts.append(user_prompt)
        user_display_message["content"] = user_prompt

    processed_image = None
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            # Konvertiere RGBA zu RGB falls nötig (manche PNGs)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            user_message_parts.append(image) # Das SDK kann PIL Images direkt verarbeiten
            user_display_message["image"] = uploaded_image # Zum Anzeigen im Chat
            st.success(f"Bild '{uploaded_image.name}' hochgeladen und wird gesendet.")
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten des Bildes: {e}")
            uploaded_image = None # Verhindere Senden bei Fehler


    # Nur fortfahren, wenn wir Text oder ein erfolgreich verarbeitetes Bild haben
    if user_message_parts:
        st.session_state.messages.append(user_display_message)
        with st.chat_message("user"):
            if "image" in user_display_message and user_display_message["image"] is not None:
                st.image(user_display_message["image"], width=300)
            if "content" in user_display_message:
                 st.markdown(user_display_message["content"])

        try:
            with st.spinner("Denke nach..."):
                if not user_message_parts: # Fallback falls user_prompt leer war und Bild fehlgeschlagen
                     if user_prompt: # Wenn nur Text da war, aber keine parts Liste
                        response = st.session_state.chat_session.send_message(user_prompt)
                     else: # Sollte nicht passieren, aber als Sicherheit
                        raise ValueError("Keine Nachricht oder Bild zum Senden vorhanden.")
                else:
                    response = st.session_state.chat_session.send_message(user_message_parts)

            assistant_response_text = response.text
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_text})
            # Clear the uploader after processing by rerunning the script
            st.session_state.file_uploader_key = str(Path(uploaded_image.name if uploaded_image else "img").stem) + str(os.urandom(4))


        except Exception as e:
            st.error(f"Ein Fehler ist bei der Kommunikation mit dem Feedback-Modell aufgetreten: {e}")
            error_message_content = f"Entschuldigung, ich bin auf einen Fehler gestoßen: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message_content})
        
        # Reset the uploader by rerunning. A bit of a hack for file_uploader.
        # A more robust way would involve managing the uploader's state more directly if possible.
        st.rerun()

elif not st.session_state.messages and st.session_state.chat_session : # Initial state, Bot hasn't said anything
     # This case is usually handled by the initial bot message logic after model init
     pass
elif not st.session_state.chat_session and st.session_state.solutions_content:
    st.info("Feedback Assistent wird initialisiert. Bitte warten...")
    st.rerun() # Trigger re-initialization if model is not loaded but content is
