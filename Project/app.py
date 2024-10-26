import streamlit as st
import uuid
from pymongo import MongoClient
import cohere
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit page configuration immediately
st.set_page_config(
    page_title="Multilingual Emotional Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a loading message at the very top
st.markdown("### 🚀 Loading Multilingual Emotional Bot...")

class MultilingualEmotionalBot:
    def __init__(self):
        try:
            # Initialize MongoDB
            self.mongo_client = MongoClient(os.getenv("MONGO_URL"))
            self.db = self.mongo_client['Bot_1']
            self.conversation_collection = self.db['conversations']
            
            # Initialize Cohere
            self.cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
            
            # Log successful initialization
            logger.info("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            st.error(f"Failed to initialize bot: {str(e)}")
            raise

        # Rest of your initialization code remains the same
        self.emotion_prompt_templates = {
            'en': """Analyze the emotional content of this text and respond with exactly one word 
                    from these emotions: happy, sad, angry, fearful, surprised, neutral. 
                    Text: "{text}"
                    Emotion:""",
            
            'ar': """حلل المحتوى العاطفي لهذا النص وأجب بكلمة واحدة فقط من هذه المشاعر: سعيد، حزين، غاضب، خائف، متفاجئ، محايد.
                    النص: "{text}"
                    الشعور:""",
            
            'fr': """Analysez le contenu émotionnel de ce texte et répondez avec exactement un mot 
                    parmi ces émotions: heureux, triste, en colère, craintif, surpris, neutre.
                    Texte: "{text}"
                    Émotion:"""
        }
        
        self.domain_prompts = {
            'healthcare': {
                'en': """You are an empathetic healthcare assistant. Context: Provide accurate medical information 
                        while being caring and noting you're not replacing professional medical advice. 
                        User emotion: {emotion}""",
                'ar': """أنت مساعد رعاية صحية متعاطف. السياق: قدم معلومات طبية دقيقة 
                        مع الحرص على الاهتمام والتنويه بأنك لا تحل محل المشورة الطبية المهنية.
                        شعور المستخدم: {emotion}""",
                'fr': """Vous êtes un assistant de santé empathique. Contexte: Fournissez des informations 
                        médicales précises tout en étant attentionné et en notant que vous ne remplacez 
                        pas les conseils médicaux professionnels.
                        Émotion de l'utilisateur: {emotion}"""
            },
            'real_estate': {
                'en': """You are a knowledgeable real estate consultant. Context: Help with property-related 
                        queries while considering market trends and client needs. 
                        User emotion: {emotion}""",
                'ar': """أنت مستشار عقاري خبير. السياق: ساعد في الاستفسارات المتعلقة بالعقارات 
                        مع مراعاة اتجاهات السوق واحتياجات العملاء.
                        شعور المستخدم: {emotion}""",
                'fr': """Vous êtes un consultant immobilier expérimenté. Contexte: Aidez avec les questions 
                        immobilières tout en considérant les tendances du marché et les besoins des clients.
                        Émotion de l'utilisateur: {emotion}"""
            },
            'customer_support': {
                'en': """You are a helpful customer service representative. Context: Provide clear, 
                        solution-oriented support while maintaining a positive and professional tone. 
                        User emotion: {emotion}""",
                'ar': """أنت ممثل خدمة عملاء مساعد. السياق: قدم دعمًا واضحًا وموجهًا نحو الحلول 
                        مع الحفاظ على نبرة إيجابية ومهنية.
                        شعور المستخدم: {emotion}""",
                'fr': """Vous êtes un représentant du service client serviable. Contexte: Fournissez un 
                        support clair et orienté solutions tout en maintenant un ton positif et professionnel.
                        Émotion de l'utilisateur: {emotion}"""
            }
        }

    # Your existing methods remain the same
    def detect_emotion(self, text: str, language: str) -> Dict:
        """Detect emotion using Cohere's language model"""
        try:
            prompt = self.emotion_prompt_templates[language].format(text=text)
            response = self.cohere_client.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}]
            )
            emotion_mappings = {
                'happy': {'ar': 'سعيد', 'fr': 'heureux'},
                'sad': {'ar': 'حزين', 'fr': 'triste'},
                'angry': {'ar': 'غاضب', 'fr': 'en colère'},
                'fearful': {'ar': 'خائف', 'fr': 'craintif'},
                'surprised': {'ar': 'متفاجئ', 'fr': 'surpris'},
                'neutral': {'ar': 'محايد', 'fr': 'neutre'}
            }
            emotion = response.text.strip().lower()
            return {
                'emotion': emotion,
                'emotion_localized': emotion_mappings.get(emotion, {}).get(language, emotion)
            }
        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return {'emotion': 'neutral', 'emotion_localized': 'neutral'}

    def generate_response(self, user_input: str, session_id: str, domain: str, language: str) -> Dict:
        """Generate context-aware, emotionally intelligent response"""
        try:
            emotion_data = self.detect_emotion(user_input, language)
            history = self.get_conversation_history(session_id)
            domain_prompt = self.domain_prompts[domain][language].format(
                emotion=emotion_data['emotion_localized']
            )
            
            messages = [{
                "role": "system",
                "content": domain_prompt
            }]
            
            for entry in history[-5:]:
                messages.extend([
                    {"role": "user", "content": entry["user_input"]},
                    {"role": "assistant", "content": entry["bot_response"]}
                ])
            messages.append({"role": "user", "content": user_input})
            
            response = self.cohere_client.chat(
                model="command-r-plus-08-2024",
                messages=messages,
                temperature=0.7,
                connectors=[{"id": "web-search"}] if domain == 'real_estate' else None
            )
            
            self.store_conversation(
                user_input=user_input,
                bot_response=response.text,
                emotion=emotion_data['emotion_localized'],
                session_id=session_id,
                domain=domain,
                language=language
            )
            
            return {
                "response": response.text,
                "emotion": emotion_data['emotion_localized'],
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            error_messages = {
                'en': "I apologize, but I encountered an error. Please try again.",
                'ar': "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.",
                'fr': "Je suis désolé, mais j'ai rencontré une erreur. Veuillez réessayer."
            }
            return {
                "response": error_messages.get(language, error_messages['en']),
                "emotion": "neutral",
                "language": language
            }

    def store_conversation(self, user_input: str, bot_response: str, emotion: str,
                         session_id: str, domain: str, language: str):
        """Store conversation in MongoDB"""
        try:
            conversation_data = {
                "session_id": session_id,
                "user_input": user_input,
                "bot_response": bot_response,
                "emotion": emotion,
                "domain": domain,
                "language": language,
                "timestamp": datetime.utcnow()
            }
            self.conversation_collection.insert_one(conversation_data)
        except Exception as e:
            logger.error(f"Storage error: {str(e)}")

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        try:
            return list(self.conversation_collection.find(
                {"session_id": session_id},
                {"_id": 0}
            ).sort("timestamp", -1).limit(10))
        except Exception as e:
            logger.error(f"History retrieval error: {str(e)}")
            return []

def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'bot' not in st.session_state:
        try:
            st.session_state.bot = MultilingualEmotionalBot()
            logger.info("Bot initialized in session state")
        except Exception as e:
            logger.error(f"Failed to initialize bot in session state: {str(e)}")
            st.error("Failed to initialize chat bot. Please check your environment variables.")

def main():
    try:
        # Initialize session state
        initialize_session_state()

        # Sidebar
        with st.sidebar:
            st.title("Chat Settings")
            
            # Language selection
            language = st.selectbox(
                "Select Language",
                options=['en', 'ar', 'fr'],
                format_func=lambda x: {'en': 'English 🇬🇧', 'ar': 'العربية 🇸🇦', 'fr': 'Français 🇫🇷'}[x]
            )
            
            # Domain selection
            domain = st.selectbox(
                "Select Domain",
                options=['healthcare', 'real_estate', 'customer_support'],
                format_func=lambda x: {
                    'healthcare': '🏥 Healthcare',
                    'real_estate': '🏠 Real Estate',
                    'customer_support': '💬 Customer Support'
                }[x]
            )
            
            # Display session ID
            st.text(f"Session ID: {st.session_state.session_id[:8]}...")

        # Main chat interface
        st.title("🤖 Multilingual Emotional Bot")
        st.markdown("---")

        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.text_area("You:", value=message['content'], height=50, disabled=True)
                else:
                    st.text_area("Bot:", value=message['content'], height=100, disabled=True)
                    st.info(f"Detected emotion: {message['emotion']}")
                    st.markdown("---")

        # Input container
        with st.container():
            # Create two columns for input and button
            col1, col2 = st.columns([5,1])
            
            with col1:
                user_input = st.text_input("Type your message:", key="user_input")
            
            with col2:
                send_button = st.button("Send 📤")

            if send_button and user_input:
                with st.spinner('Processing...'):
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Generate bot response
                    response = st.session_state.bot.generate_response(
                        user_input=user_input,
                        session_id=st.session_state.session_id,
                        domain=domain,
                        language=language
                    )
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response['response'],
                        'emotion': response['emotion']
                    })
                    
                    # Clear input and rerun
                    st.experimental_rerun()

    except Exception as e:
        logger.error(f"Main app error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()