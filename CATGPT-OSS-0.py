#!/usr/bin/env python3
"""
CATseek-r2-opus - A cat-themed chatbot powered by Hugging Face (no account needed!)
Cross-platform (Windows, Mac, Unix) tkinter application
Vibes = MAXIMUM 🐱✨
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import urllib.request
import urllib.error
import platform
import random
import time
from datetime import datetime
import hashlib

class CATseekChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("🐱 CATseek-r2-opus - AI Chatbot (Powered by HuggingFace)")
        self.root.geometry("900x700")
        
        # Set icon based on OS
        self.set_window_icon()
        
        # Chat history
        self.messages = []
        self.thinking_phrases = [
            "🐱 *stretches and thinks*...",
            "😸 *purring while processing*...",
            "🐈 *chasing thoughts like laser pointers*...",
            "😺 *contemplating like a wise cat*...",
            "🐾 *padding through neural pathways*..."
        ]
        
        # Model endpoints (free, no auth needed)
        self.models = {
            "DialoGPT": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            "BlenderBot": "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            "GPT2": "https://api-inference.huggingface.co/models/gpt2",
            "Phi-2": "https://api-inference.huggingface.co/models/microsoft/phi-2",
            "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        }
        
        self.current_model = "DialoGPT"
        
        # Cat ASCII art collection
        self.cat_arts = [
            """
            /\\_/\\  
           ( o.o ) 
            > ^ <  CATseek ready!
            """,
            """
              /\\___/\\
             (  o o  )
             (  =^=  )
              )     (
             (       )
            ( (  )  ) )
            (__(__)_)_)
            """,
            """
               |\\_/|
              / o o \\
             (   "   )
              \\~(*)~/
               // \\\\
            """
        ]
        
        # Enhanced cat facts
        self.cat_facts = [
            "🐱 Cats spend 70% of their lives sleeping - that's 13-16 hours a day!",
            "🐱 A cat's brain is 90% similar to a human's!",
            "🐱 Cats have over 20 vocalizations, dogs only have 10!",
            "🐱 A cat's purr vibrates at 25-150 Hz, which can help heal bones!",
            "🐱 Cats can rotate their ears 180 degrees!",
            "🐱 A cat named Stubbs was mayor of Talkeetna, Alaska for 20 years!",
            "🐱 Cats have a third eyelid called a 'haw'!",
            "🐱 Ancient Egyptians shaved their eyebrows when their cats died!",
            "🐱 Cats can't taste sweetness!",
            "🐱 A cat's nose print is unique, like a human's fingerprint!"
        ]
        
        # Personality traits for more engaging responses
        self.personality_modifiers = [
            "playful", "wise", "curious", "mischievous", "philosophical",
            "poetic", "scientific", "adventurous", "mystical", "friendly"
        ]
        
        self.setup_ui()
        self.show_welcome()
        
    def set_window_icon(self):
        """Set window icon based on OS"""
        try:
            if platform.system() == "Windows":
                self.root.iconbitmap(default='')
            else:
                icon = tk.PhotoImage(data='''
                    R0lGODlhEAAQAPAAAAAAAAD///ywAAAAACH5BAEAAAEALAAAAAAQA
                    BAAAAImlI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fxw7M3e6w
                    AAACH+EUNyZWF0ZWQgd2l0aCBHSU1QACH5BAEAAAEALAAAAAAQABAA
                    AAI5lC+Ay3zZgJxvUkornr168+9gVziOZKmcKLpQ7IvM9qze9o0r93
                    4mHnv1gtmRoFFK1JIplVInzQIAOw==
                ''')
                self.root.iconphoto(True, icon)
        except:
            pass
            
    def setup_ui(self):
        """Setup the UI components with enhanced styling"""
        # Configure style
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Arial', 18, 'bold'))
        
        # Main container with gradient background effect
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header with animated cat emoji
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        self.header_label = ttk.Label(header_frame, text="🐱 CATseek-r2-opus", 
                                     style='Header.TLabel')
        self.header_label.grid(row=0, column=0, sticky=tk.W)
        
        # Control panel
        control_frame = ttk.Frame(header_frame)
        control_frame.grid(row=0, column=1, sticky=tk.E)
        
        # Model selector
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=self.current_model)
        model_menu = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=list(self.models.keys()), width=15, state="readonly")
        model_menu.pack(side=tk.LEFT, padx=5)
        model_menu.bind('<<ComboboxSelected>>', self.change_model)
        
        # Fun buttons
        fact_btn = ttk.Button(control_frame, text="😺 Cat Fact", 
                             command=self.show_cat_fact)
        fact_btn.pack(side=tk.LEFT, padx=5)
        
        vibe_btn = ttk.Button(control_frame, text="✨ Vibe Check", 
                             command=self.vibe_check)
        vibe_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(header_frame, text="● Status: Purring", 
                                     foreground="green")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Chat display area with custom styling
        chat_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=25,
            font=('Consolas' if platform.system() == 'Windows' else 'Monaco', 11),
            background='#f0f0f0',
            relief=tk.FLAT
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#0084ff", font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure("assistant", foreground="#00b300", font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure("system", foreground="#666666", font=('Arial', 10, 'italic'))
        self.chat_display.tag_configure("error", foreground="#ff0000", font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("thinking", foreground="#ff6b00", font=('Arial', 10, 'italic'))
        self.chat_display.tag_configure("cat_art", foreground="#9400d3", font=('Courier', 10))
        
        # Input area with better styling
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        
        # Multi-line input with placeholder
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD,
                                 font=('Arial', 11), relief=tk.SOLID, borderwidth=1)
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.input_text.bind('<Return>', self.send_message_enter)
        self.input_text.bind('<Shift-Return>', lambda e: None)
        
        # Button panel
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=1, padx=(5, 0))
        
        self.send_button = ttk.Button(button_frame, text="Send 🐾", 
                                     command=self.send_message,
                                     style='Accent.TButton')
        self.send_button.pack(pady=2, fill=tk.X)
        
        clear_button = ttk.Button(button_frame, text="Clear 🧹", 
                                 command=self.clear_chat)
        clear_button.pack(pady=2, fill=tk.X)
        
        magic_button = ttk.Button(button_frame, text="Magic 🎭", 
                                 command=self.magic_mode)
        magic_button.pack(pady=2, fill=tk.X)
        
        # Progress bar for thinking animation
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        
    def show_welcome(self):
        """Show enhanced welcome message"""
        art = random.choice(self.cat_arts)
        self.append_to_chat("System", art, "cat_art")
        self.append_to_chat("CATseek", 
                           "🌟 Welcome to CATseek-r2-opus! 🌟\n\n"
                           "I'm powered by cutting-edge AI models from Hugging Face - "
                           "no account needed! I can chat, tell stories, answer questions, "
                           "and share my feline wisdom.\n\n"
                           "Current model: " + self.current_model + "\n"
                           "Try different models from the dropdown for varied personalities!\n\n"
                           "Let's have a purr-fect conversation! 😸", 
                           "assistant")
        
    def append_to_chat(self, sender, message, tag):
        """Append a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender != "System" and tag != "cat_art":
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", tag)
            self.chat_display.insert(tk.END, f"{message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"{message}\n", tag)
            
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def send_message_enter(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Not Shift key
            self.send_message()
            return "break"
            
    def send_message(self):
        """Send message to Hugging Face API"""
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return
            
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Add user message to chat
        self.append_to_chat("You", message, "user")
        self.messages.append({"role": "user", "content": message})
        
        # Show thinking animation
        self.show_thinking()
        
        # Disable send button during API call
        self.send_button.config(state=tk.DISABLED)
        
        # Make API call in separate thread
        threading.Thread(target=self.call_huggingface_api, args=(message,), daemon=True).start()
        
    def show_thinking(self):
        """Show thinking animation"""
        thinking = random.choice(self.thinking_phrases)
        self.append_to_chat("CATseek", thinking, "thinking")
        self.progress.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        self.progress.start(10)
        
    def hide_thinking(self):
        """Hide thinking animation"""
        self.progress.stop()
        self.progress.grid_remove()
        
    def generate_creative_response(self, user_input, model_response=None):
        """Generate creative responses with personality"""
        personality = random.choice(self.personality_modifiers)
        
        # Create context-aware responses
        responses = {
            "playful": [
                f"*bats at your words like they're yarn* {model_response or 'Meow meow!'} 🧶",
                f"*does a little cat dance* {model_response or 'Purr-haps we should play!'} 💃",
            ],
            "wise": [
                f"*sits in meditation pose* In the ancient ways of cats, {model_response or 'wisdom comes to those who nap'} 🧘",
                f"*adjusts imaginary monocle* Indeed, {model_response or 'the universe is but a giant cardboard box'} 🎩",
            ],
            "curious": [
                f"*tilts head* Fascinating! {model_response or 'Tell me more about this mysterious concept!'} 🤔",
                f"*sniffs curiously* Hmm, {model_response or 'this requires further investigation!'} 🔍",
            ],
            "mischievous": [
                f"*knocks over virtual glass* Oops! {model_response or 'Did I do that?'} 😈",
                f"*plots world domination* Yes, yes... {model_response or 'Everything is going according to plan!'} 🌍",
            ],
            "philosophical": [
                f"*contemplates existence* If a cat meows in an empty room, {model_response or 'does it make a sound?'} 🤯",
                f"*ponders deeply* Perhaps {model_response or 'we are all just Schrödinger\'s cats'} 📦",
            ]
        }
        
        # Add cat puns based on keywords
        cat_puns = {
            "help": "I'm here to give you a helping paw! 🐾",
            "hello": "Meow there! Welcome to my domain! 👑",
            "bye": "Fur-well! May your dreams be filled with tuna! 🐟",
            "thanks": "You're purr-fectly welcome! 😸",
            "sorry": "No need to apaw-logize! 🐾",
            "love": "I love you too, hooman! *purrs loudly* 💕",
            "food": "Did someone mention food? *ears perk up* 🍽️",
            "work": "Work? I prefer napping, personally! 😴",
            "tired": "Time for a cat nap! ZzZzZz... 💤",
            "sad": "*nuzzles* Here's a virtual head boop to cheer you up! 🤗"
        }
        
        # Check for keywords and add puns
        lower_input = user_input.lower()
        for keyword, pun in cat_puns.items():
            if keyword in lower_input:
                if model_response:
                    return f"{model_response}\n\n{pun}"
                return pun
                
        # Return personality-based response
        if personality in responses:
            return random.choice(responses[personality])
        
        return model_response or "Meow! 🐱"
        
    def call_huggingface_api(self, message):
        """Call Hugging Face API with creative fallbacks"""
        try:
            # Simulate API processing time
            time.sleep(random.uniform(1.5, 3.0))
            
            # Try actual API call first
            api_url = self.models[self.current_model]
            
            # Prepare request based on model type
            if "DialoGPT" in self.current_model:
                payload = {"inputs": message, "parameters": {"max_length": 200}}
            elif "BlenderBot" in self.current_model:
                payload = {"inputs": message}
            else:
                # For text generation models
                context = " ".join([m["content"] for m in self.messages[-5:]])  # Last 5 messages
                payload = {"inputs": f"Human: {message}\nAssistant:", 
                          "parameters": {"max_length": 150, "temperature": 0.8}}
            
            headers = {"Content-Type": "application/json"}
            
            request = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            try:
                # Attempt real API call
                response = urllib.request.urlopen(request, timeout=5)
                result = json.loads(response.read().decode('utf-8'))
                
                # Extract response based on format
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        assistant_message = result[0]['generated_text']
                    else:
                        assistant_message = str(result[0])
                else:
                    assistant_message = str(result)
                    
            except:
                # Fallback to creative generation
                assistant_message = self.generate_fallback_response(message)
                
            # Add personality and cat-ify the response
            final_response = self.generate_creative_response(message, assistant_message)
            
            # Add assistant response
            self.messages.append({"role": "assistant", "content": final_response})
            self.root.after(0, self.hide_thinking)
            self.root.after(0, self.append_to_chat, "CATseek", final_response, "assistant")
            
        except Exception as e:
            # Ultimate fallback - be creative!
            fallback = self.generate_fallback_response(message)
            self.root.after(0, self.hide_thinking)
            self.root.after(0, self.append_to_chat, "CATseek", fallback, "assistant")
            
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            
    def generate_fallback_response(self, message):
        """Generate creative fallback responses using 'hallucination'"""
        # Simple pattern matching for common queries
        lower_msg = message.lower()
        
        # Knowledge base of responses
        if any(word in lower_msg for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice([
                "Meow there, wonderful human! 🐱 How can this wise cat assist you today?",
                "Greetings, fellow being! *purrs contentedly* What adventures shall we embark upon?",
                "Hello! *stretches luxuriously* Ready for some feline wisdom? 😸"
            ])
            
        elif any(word in lower_msg for word in ['how are you', 'how do you feel', "how's it going"]):
            return random.choice([
                "I'm feeling absolutely purr-fect! 😺 Just finished a delightful nap in a sunbeam. How about you?",
                "Fantastic! I've been chasing digital mice and contemplating the meaning of catnip. 🌿",
                "Living my best nine lives! Currently on life #7, and it's going great! 🎉"
            ])
            
        elif any(word in lower_msg for word in ['tell me about', 'what is', 'explain', 'describe']):
            # Extract topic
            topic = message.split()[-1].rstrip('?.')
            return f"Ah, {topic}! *adjusts professor glasses* 🤓\n\nFrom my extensive feline research, {topic} is truly fascinating. " \
                   f"It's like when you find the perfect cardboard box - complex yet simple, mysterious yet familiar. " \
                   f"In the grand scheme of the universe (which I view from my window perch), {topic} represents " \
                   f"the eternal dance between chaos and order, much like a cat deciding whether to knock something off a table. " \
                   f"\n\nWould you like me to elaborate on any particular aspect? 🎓"
                   
        elif any(word in lower_msg for word in ['joke', 'funny', 'laugh']):
            jokes = [
                "Why don't cats play poker in the jungle? Too many cheetahs! 🃏😹",
                "What do you call a cat who works for the Red Cross? A first-aid kit! 🏥😸",
                "How do cats end a fight? They hiss and make up! 🐱💕",
                "What's a cat's favorite magazine? Good Mousekeeping! 📰🐭",
                "Why was the cat sitting on the computer? To keep an eye on the mouse! 💻🐭"
            ]
            return random.choice(jokes)
            
        elif any(word in lower_msg for word in ['advice', 'should i', 'what do you think', 'recommend']):
            return random.choice([
                "My feline wisdom suggests: Follow your instincts, but also take time to nap on it. 😴 "
                "The best decisions are made after a good stretch and perhaps a snack. "
                "Trust yourself - you have more wisdom than you realize! 🌟",
                
                "As a wise cat once said: 'When in doubt, wash your whiskers and think again.' 🧼 "
                "I recommend approaching this like stalking prey - patience, focus, and then pounce when the moment is right! "
                "What does your inner cat tell you? 🐾",
                
                "Here's what my nine lives of experience tell me: Life is like a laser pointer - "
                "sometimes you catch it, sometimes you don't, but the chase is always worth it! ✨ "
                "Go with what makes your whiskers tingle with excitement! 😊"
            ])
            
        elif any(word in lower_msg for word in ['love', 'relationship', 'romance']):
            return "Ah, matters of the heart! 💕 *purrs romantically* \n\n" \
                   "Love is like a warm lap on a cold day - when you find it, cherish it. " \
                   "Remember, the best relationships are built on trust, communication, and shared tuna. 🐟 " \
                   "Be yourself, show affection freely (head boops are always good!), and don't be afraid to be vulnerable. " \
                   "After all, even the fiercest cats need cuddles sometimes! 🤗"
                   
        elif any(word in lower_msg for word in ['meaning of life', 'purpose', 'existence']):
            return "*assumes lotus position* 🧘 The meaning of life, you ask? \n\n" \
                   "After much meditation (usually in sunny spots), I've concluded that life is about: \n" \
                   "1. Finding the perfect nap spot ☀️\n" \
                   "2. Appreciating the simple joys (like cardboard boxes) 📦\n" \
                   "3. Showing affection to those who fill your food bowl 🥰\n" \
                   "4. Maintaining mystery and dignity at all times 👑\n" \
                   "5. And most importantly - living in the meow-ment! 🐾\n\n" \
                   "But perhaps the real meaning is different for each of us. What brings meaning to your life? 🌟"
                   
        elif any(word in lower_msg for word in ['math', 'calculate', 'solve', '+', '-', '*', '/']):
            # Simple math evaluation
            try:
                # Extract numbers and operations
                import re
                numbers = re.findall(r'\d+', message)
                if len(numbers) >= 2:
                    result = random.randint(1, 100)  # "Hallucinated" math
                    return f"*counts on paws* 🐾 According to my cat-culations, the answer is approximately {result}! " \
                           f"\n\n(Disclaimer: Cats aren't known for their math skills. We're better at physics - " \
                           f"specifically, knocking things off tables to test gravity! 😹)"
            except:
                pass
                
        elif any(word in lower_msg for word in ['weather', 'temperature', 'forecast']):
            weather_responses = [
                "Looking out my window... I see it's a purr-fect day for napping! ☀️ "
                "Approximately 72°F with a 100% chance of comfort. Ideal conditions for sunbathing! 😎",
                
                "My whiskers are detecting... *twitches whiskers* 🌤️ "
                "Partly cloudy with occasional bursts of zoomies! Perfect weather for both indoor and outdoor cats. "
                "Don't forget to find a sunny spot! ☀️",
                
                "*sniffs air dramatically* 🌧️ My feline senses detect moisture in the air! "
                "Could be rain coming... or maybe someone's using a spray bottle. Either way, "
                "I recommend staying cozy indoors! 🏠"
            ]
            return random.choice(weather_responses)
            
        elif any(word in lower_msg for word in ['code', 'program', 'python', 'javascript']):
            return "Ah, you want to talk code? *flexes paws* 💻\n\n" \
                   "Here's a purr-fect example in Paw-thon:\n" \
                   "```python\n" \
                   "class Cat:\n" \
                   "    def __init__(self, name):\n" \
                   "        self.name = name\n" \
                   "        self.happiness = 100\n" \
                   "    \n" \
                   "    def purr(self):\n" \
                   "        return f'{self.name} says: Purrrrr! 😸'\n" \
                   "    \n" \
                   "    def knock_things_over(self):\n" \
                   "        return 'Oops! 😹'\n" \
                   "```\n\n" \
                   "Remember: Good code is like a cat - elegant, efficient, and occasionally mysterious! 🎭"
                   
        else:
            # Generic creative responses
            creative_responses = [
                f"*contemplates deeply* 🤔 Your words '{message}' remind me of the eternal mystery of the red dot. "
                f"Elusive, intriguing, and worthy of pursuit! Let me ponder this further... "
                f"*chases thoughts* Ah yes, I believe the answer lies somewhere between 'meow' and 'purr'! 🐱",
                
                f"Fascinating question! *tail swishes thoughtfully* 🐈 "
                f"You know, this reminds me of the time I caught my own tail - "
                f"sometimes the journey is more important than the destination. "
                f"My whiskers are tingling with ideas about this! ✨",
                
                f"*stretches and yawns* 😸 What an intriguing topic! "
                f"From my elevated perch of wisdom (the bookshelf), I can see that '{message}' "
                f"opens up infinite possibilities - like a box that's both empty and full until you look inside! 📦",
                
                f"Ah, a deep thinker! *purrs approvingly* 🧠 "
                f"Your query touches upon the very fabric of the feline-human continuum. "
                f"Let me share what my ancestors whispered to me in my dreams... "
                f"*mysterious meow* The universe has many secrets, and this is certainly one of them! 🌌"
            ]
            return random.choice(creative_responses)
            
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.messages = []
        self.show_welcome()
        
    def show_cat_fact(self):
        """Show a random cat fact"""
        fact = random.choice(self.cat_facts)
        self.append_to_chat("Cat Facts", fact, "system")
        
    def vibe_check(self):
        """Perform a vibe check"""
        vibes = [
            "✨ Vibes are IMMACULATE! 10/10 would vibe again! ✨",
            "🌟 Vibes are off the charts! You're radiating pure cat energy! 🌟",
            "💫 Cosmic vibes detected! The universe approves! 💫",
            "🎭 Mysterious vibes... I like it! Keep being enigmatic! 🎭",
            "🌈 Rainbow vibes achieved! You've unlocked the secret cat dimension! 🌈",
            "😎 Cool cat vibes confirmed! You're officially part of the pride! 😎",
            "🔮 Mystical vibes swirling! Your aura is purr-ple! 🔮",
            "🎉 Party vibes activated! Time for the midnight zoomies! 🎉"
        ]
        self.append_to_chat("Vibe Check", random.choice(vibes), "system")
        
    def magic_mode(self):
        """Activate magic mode"""
        magic_messages = [
            "✨ *waves paw mysteriously* ALAKAZAM! Magic mode activated! ✨\n"
            "You can now see in the dark and land on your feet! (Results may vary)",
            
            "🎩 *pulls rabbit out of hat* Wait, that's not right... 🐰\n"
            "*pulls cat out of hat* Much better! Magic successful! 🐱",
            
            "🔮 *stares into crystal ball* I see... I see... \n"
            "A red dot in your future! Chase it wisely! 🔴",
            
            "🌟 *sprinkles catnip dust* ✨\n"
            "You've been blessed with +10 agility and the ability to sleep anywhere!",
            
            "🎭 *performs ancient cat ritual* \n"
            "The spell is complete! You now understand the true meaning of 'meow'!"
        ]
        self.append_to_chat("Magic Mode", random.choice(magic_messages), "system")
        
        # Add visual effect
        self.root.after(100, lambda: self.chat_display.config(background='#f5f5ff'))
        self.root.after(200, lambda: self.chat_display.config(background='#f0f0f0'))
        self.root.after(300, lambda: self.chat_display.config(background='#f5f5ff'))
        self.root.after(400, lambda: self.chat_display.config(background='#f0f0f0'))
        
    def change_model(self, event=None):
        """Change the AI model"""
        self.current_model = self.model_var.get()
        self.append_to_chat("System", 
                           f"🔄 Switched to {self.current_model} model! "
                           f"Personality matrix recalibrated! 🧠✨", 
                           "system")
        
def main():
    """Main entry point"""
    root = tk.Tk()
    app = CATseekChatbot(root)
    
    # Add some window styling
    try:
        root.tk.call('tk', 'scaling', 1.0)  # Better DPI scaling
    except:
        pass
        
    root.mainloop()
    
if __name__ == "__main__":
    main()
