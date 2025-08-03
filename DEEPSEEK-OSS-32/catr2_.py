#!/usr/bin/env python3
"""
CATSEEK R2 - Advanced AI Chat Application
Optimized for Intel NPU with DeepSeek-R1-Zero
Windows Tkinter Application for Python 3.13
"""

import sys
import os
import subprocess
import threading
import queue
import json
import time
import platform
from datetime import datetime
from pathlib import Path

# Standard library imports that should always work
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import webbrowser

# ─── Auto-Installer for Dependencies ─────────────────────
class DependencyInstaller:
    def __init__(self, parent=None):
        self.parent = parent
        self.required_packages = {
            'torch': 'torch',
            'transformers': 'transformers>=4.40.0',
            'accelerate': 'accelerate>=0.27.0',
            'safetensors': 'safetensors>=0.4.3',
            'huggingface-hub': 'huggingface-hub',
            'Pillow': 'pillow',
            'psutil': 'psutil',
            'bitsandbytes': 'bitsandbytes',  # For 4-bit quantization
            'intel-extension-for-pytorch': 'intel-extension-for-pytorch',  # Intel optimization
            'openvino': 'openvino>=2024.0.0',  # Intel OpenVINO for NPU
            'openvino-dev': 'openvino-dev[pytorch]',  # OpenVINO development tools
            'optimum': 'optimum[openvino,nncf]',  # Hugging Face Optimum for Intel
        }
        
    def check_package(self, package_name):
        try:
            if package_name == 'intel-extension-for-pytorch':
                __import__('intel_extension_for_pytorch')
            else:
                __import__(package_name)
            return True
        except ImportError:
            return False
    
    def install_packages(self, progress_callback=None):
        missing = []
        for pkg, install_name in self.required_packages.items():
            if not self.check_package(pkg):
                missing.append((pkg, install_name))
        
        if not missing:
            return True
        
        total = len(missing)
        for i, (pkg, install_name) in enumerate(missing):
            if progress_callback:
                progress_callback(f"Installing {pkg}...", (i / total) * 100)
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", install_name
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                # Try without version constraints if it fails
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", pkg
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    # Intel packages might fail on non-Intel systems, continue anyway
                    if 'intel' not in pkg and 'openvino' not in pkg:
                        if progress_callback:
                            progress_callback(f"Failed to install {pkg}", -1)
                        return False
        
        if progress_callback:
            progress_callback("All dependencies installed!", 100)
        return True

# ─── Intel Hardware Detection ─────────────────────
class IntelHardwareDetector:
    @staticmethod
    def detect_intel_cpu():
        """Detect if running on Intel CPU"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'intel' in info.get('vendor_id_raw', '').lower()
        except:
            # Fallback to platform detection
            return 'intel' in platform.processor().lower()
    
    @staticmethod
    def detect_intel_gpu():
        """Detect Intel GPU/NPU availability"""
        try:
            import subprocess
            # Try to detect Intel GPU using OpenVINO
            result = subprocess.run(['openvino_device_query'], capture_output=True, text=True)
            return 'GPU' in result.stdout or 'NPU' in result.stdout
        except:
            return False
    
    @staticmethod
    def get_best_device():
        """Get the best available Intel device"""
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            
            # Priority: NPU > GPU > CPU
            if 'NPU' in devices:
                return 'NPU'
            elif 'GPU' in devices:
                return 'GPU'
            elif 'CPU' in devices:
                return 'CPU'
            else:
                return 'AUTO'  # Let OpenVINO choose
        except:
            return None

# ─── Loading Screen ─────────────────────
class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CATSEEK R2 - Loading")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.root.winfo_screenheight() // 2) - (300 // 2)
        self.root.geometry(f"500x300+{x}+{y}")
        
        # Dark theme
        self.root.configure(bg='#1e1e1e')
        
        # Logo/Title
        title = tk.Label(
            self.root,
            text="🐱 CATSEEK R2",
            font=("Arial", 24, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title.pack(pady=30)
        
        # Subtitle
        subtitle = tk.Label(
            self.root,
            text="Intel NPU Optimized Edition",
            font=("Arial", 10),
            bg='#1e1e1e',
            fg='#00aaff'
        )
        subtitle.pack(pady=(0, 20))
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Initializing...",
            font=("Arial", 12),
            bg='#1e1e1e',
            fg='#cccccc'
        )
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            length=400,
            mode='determinate'
        )
        self.progress.pack(pady=20)
        
        # Details label
        self.details_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            bg='#1e1e1e',
            fg='#888888'
        )
        self.details_label.pack(pady=5)
        
    def update_status(self, message, progress=-1, details=""):
        self.status_label.config(text=message)
        if progress >= 0:
            self.progress['value'] = progress
        else:
            self.progress.config(mode='indeterminate')
            self.progress.start(10)
        if details:
            self.details_label.config(text=details)
        self.root.update()
    
    def close(self):
        self.root.destroy()

# ─── Main CATSEEK R2 Application ─────────────────────
class CATSEEKr2:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CATSEEK R2 - AI Assistant (Intel NPU Optimized)")
        self.root.geometry("1200x800")
        
        # Application state
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.optimized_model = None
        self.intel_device = None
        self.conversation_history = []
        self.is_generating = False
        self.current_model_name = "DeepSeek-R1-Zero"
        
        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        
        # Start model loading in background
        self.model_thread = threading.Thread(target=self.load_deepseek_r1_zero, daemon=True)
        self.model_thread.start()
        
        # Start message queue processor
        self.root.after(100, self.process_message_queue)
        
    def setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Chat history and settings
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Chat history section
        history_label = ttk.Label(left_panel, text="Chat History", font=("Arial", 12, "bold"))
        history_label.pack(pady=(0, 5))
        
        # Chat list
        self.chat_listbox = tk.Listbox(left_panel, height=15)
        self.chat_listbox.pack(fill=tk.BOTH, expand=True)
        self.chat_listbox.insert(0, "New Chat 1")
        self.chat_listbox.selection_set(0)
        
        # New chat button
        new_chat_btn = ttk.Button(left_panel, text="➕ New Chat", command=self.new_chat)
        new_chat_btn.pack(pady=10, fill=tk.X)
        
        # Model info section
        model_frame = ttk.LabelFrame(left_panel, text="Model Information", padding=10)
        model_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.model_label = ttk.Label(model_frame, text=f"Model: {self.current_model_name}")
        self.model_label.pack(anchor=tk.W)
        
        self.device_label = ttk.Label(model_frame, text="Device: Detecting...")
        self.device_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(model_frame, text="Status: Loading...")
        self.status_label.pack(anchor=tk.W)
        
        # Right panel - Chat interface
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chat display
        chat_frame = ttk.Frame(right_panel)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            state=tk.DISABLED,
            bg="#f5f5f5"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for message styling
        self.chat_display.tag_configure("user", foreground="#0066cc", font=("Arial", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#009900", font=("Arial", 11, "bold"))
        self.chat_display.tag_configure("system", foreground="#666666", font=("Arial", 10, "italic"))
        
        # Input area
        input_frame = ttk.Frame(right_panel)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Input text field
        self.input_text = tk.Text(input_frame, height=3, font=("Arial", 11))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_text.bind("<Return>", self.on_enter_pressed)
        self.input_text.bind("<Control-Return>", lambda e: self.input_text.insert(tk.INSERT, "\n"))
        
        # Send button
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.send_button = ttk.Button(
            button_frame,
            text="Send 📤",
            command=self.send_message,
            state=tk.DISABLED
        )
        self.send_button.pack(pady=(0, 5))
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop ⏹️",
            command=self.stop_generation,
            state=tk.DISABLED
        )
        self.stop_button.pack()
        
        # Menu bar
        self.create_menu()
        
        # Welcome message
        self.add_message("system", "Welcome to CATSEEK R2! Loading DeepSeek-R1-Zero with Intel NPU optimizations...")
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Chat", command=self.new_chat)
        file_menu.add_command(label="Save Chat", command=self.save_chat)
        file_menu.add_command(label="Load Chat", command=self.load_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Model Settings", command=self.show_model_settings)
        settings_menu.add_command(label="Intel NPU Info", command=self.show_intel_info)
        settings_menu.add_command(label="Clear Cache", command=self.clear_cache)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
    
    def load_deepseek_r1_zero(self):
        """Load DeepSeek-R1-Zero with Intel NPU optimizations"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            self.message_queue.put(("status", "Loading DeepSeek-R1-Zero..."))
            
            # Detect Intel hardware
            detector = IntelHardwareDetector()
            is_intel = detector.detect_intel_cpu()
            self.intel_device = detector.get_best_device()
            
            if self.intel_device:
                self.message_queue.put(("device", f"Intel {self.intel_device} detected"))
            
            # Load tokenizer
            self.message_queue.put(("status", "Loading tokenizer..."))
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Zero")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine optimal loading strategy
            use_openvino = False
            use_ipex = False
            
            # Try Intel Extension for PyTorch first
            if is_intel:
                try:
                    import intel_extension_for_pytorch as ipex
                    use_ipex = True
                    self.message_queue.put(("status", "Using Intel Extension for PyTorch..."))
                except ImportError:
                    pass
            
            # Try OpenVINO for NPU/GPU
            if self.intel_device in ['NPU', 'GPU']:
                try:
                    from optimum.intel import OVModelForCausalLM
                    use_openvino = True
                    self.message_queue.put(("status", f"Using OpenVINO for Intel {self.intel_device}..."))
                except ImportError:
                    pass
            
            # Load model with appropriate optimization
            if use_openvino:
                # Use OpenVINO optimized model
                try:
                    self.model = OVModelForCausalLM.from_pretrained(
                        "deepseek-ai/DeepSeek-R1-Zero",
                        export=True,
                        device=self.intel_device,
                        ov_config={"PERFORMANCE_HINT": "LATENCY"},
                        compile=True
                    )
                    self.message_queue.put(("status", f"Model optimized for Intel {self.intel_device}"))
                except Exception as e:
                    self.message_queue.put(("status", "OpenVINO failed, falling back to PyTorch..."))
                    use_openvino = False
            
            if not use_openvino:
                # Load with PyTorch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Model loading configuration
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if is_intel else torch.float16,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
                
                # Add acceleration options
                try:
                    import accelerate
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    pass
                
                # Try 4-bit quantization for memory efficiency
                try:
                    import bitsandbytes
                    if device == "cpu":
                        # Don't use 4-bit on CPU
                        pass
                    else:
                        model_kwargs["load_in_4bit"] = True
                        self.message_queue.put(("status", "Using 4-bit quantization..."))
                except ImportError:
                    pass
                
                # Load the model
                self.message_queue.put(("status", "Loading model weights..."))
                self.model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-R1-Zero",
                    **model_kwargs
                )
                
                # Apply Intel optimizations if available
                if use_ipex:
                    import intel_extension_for_pytorch as ipex
                    self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
                    self.message_queue.put(("status", "Applied Intel CPU optimizations"))
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() and not use_openvino else -1
            )
            
            self.message_queue.put(("model_loaded", "DeepSeek-R1-Zero"))
            
        except Exception as e:
            self.message_queue.put(("error", str(e)))
    
    def process_message_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == "status":
                    self.status_label.config(text=f"Status: {msg_data}")
                elif msg_type == "device":
                    self.device_label.config(text=f"Device: {msg_data}")
                elif msg_type == "model_loaded":
                    self.model_label.config(text=f"Model: {msg_data}")
                    self.status_label.config(text="Status: Ready")
                    self.send_button.config(state=tk.NORMAL)
                    device_info = f"Intel {self.intel_device}" if self.intel_device else "CPU"
                    self.add_message("system", f"✅ DeepSeek-R1-Zero loaded successfully on {device_info}! You can now start chatting.")
                elif msg_type == "error":
                    self.status_label.config(text="Status: Error")
                    self.add_message("system", f"❌ Error: {msg_data}")
                    messagebox.showerror("Model Loading Error", f"Failed to load DeepSeek-R1-Zero:\n{msg_data}")
                elif msg_type == "assistant_message":
                    self.add_message("assistant", msg_data)
                    self.is_generating = False
                    self.send_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_message_queue)
    
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add sender label
        if sender == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] You: ", "user")
        elif sender == "assistant":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] CATSEEK: ", "assistant")
        else:  # system
            self.chat_display.insert(tk.END, f"\n[{timestamp}] System: ", "system")
        
        # Add message text
        self.chat_display.insert(tk.END, message + "\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Store in conversation history
        if sender != "system":
            self.conversation_history.append({
                "role": "user" if sender == "user" else "assistant",
                "content": message,
                "timestamp": timestamp
            })
    
    def send_message(self):
        """Send user message and generate response"""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message or self.is_generating:
            return
        
        # Clear input
        self.input_text.delete("1.0", tk.END)
        
        # Add user message
        self.add_message("user", message)
        
        # Update UI state
        self.is_generating = True
        self.send_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Generate response in background
        thread = threading.Thread(
            target=self.generate_response,
            args=(message,),
            daemon=True
        )
        thread.start()
    
    def generate_response(self, user_message):
        """Generate AI response with DeepSeek-R1-Zero"""
        try:
            if self.pipeline is None:
                self.message_queue.put(("assistant_message", "DeepSeek-R1-Zero is still loading. Please wait..."))
                return
            
            # Build conversation context
            context = ""
            for msg in self.conversation_history[-10:]:  # Last 10 messages
                role = "Human" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
            
            # Add current message
            prompt = f"{context}Human: {user_message}\nAssistant:"
            
            # Generate response with optimized parameters for Intel
            generation_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.95,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add Intel-specific optimizations
            if self.intel_device:
                generation_kwargs["num_beams"] = 1  # Greedy decoding is faster on NPU
            
            outputs = self.pipeline(prompt, **generation_kwargs)
            
            # Extract response
            generated = outputs[0]['generated_text']
            response = generated.split("Assistant:")[-1].strip()
            
            # Clean up response
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            
            # Send response to UI
            self.message_queue.put(("assistant_message", response))
            
        except Exception as e:
            self.message_queue.put(("assistant_message", f"Error generating response: {str(e)}"))
    
    def stop_generation(self):
        """Stop the current generation"""
        self.is_generating = False
        self.send_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Not Ctrl pressed
            self.send_message()
            return "break"
    
    def new_chat(self):
        """Start a new chat"""
        self.conversation_history = []
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Add to chat list
        chat_count = self.chat_listbox.size()
        self.chat_listbox.insert(tk.END, f"New Chat {chat_count + 1}")
        self.chat_listbox.selection_clear(0, tk.END)
        self.chat_listbox.selection_set(tk.END)
        
        self.add_message("system", "Started a new chat. How can I help you today?")
    
    def save_chat(self):
        """Save current chat to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", "Chat saved successfully!")
    
    def load_chat(self):
        """Load chat from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            
            # Rebuild chat display
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            
            for msg in self.conversation_history:
                role = "user" if msg["role"] == "user" else "assistant"
                self.add_message(role, msg["content"])
            
            self.chat_display.config(state=tk.DISABLED)
            messagebox.showinfo("Success", "Chat loaded successfully!")
    
    def show_model_settings(self):
        """Show model settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Model Settings")
        settings_window.geometry("500x400")
        
        # Model info
        info_frame = ttk.Frame(settings_window, padding=20)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(info_frame, text="Current Model:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(info_frame, text="DeepSeek-R1-Zero").grid(row=0, column=1, sticky=tk.W, pady=5)
        
        try:
            import torch
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            if self.intel_device:
                device = f"Intel {self.intel_device}"
            ttk.Label(info_frame, text="Device:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Label(info_frame, text=device).grid(row=1, column=1, sticky=tk.W, pady=5)
        except:
            pass
        
        # Memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            mem_text = f"{memory.used / (1024**3):.1f} / {memory.total / (1024**3):.1f} GB"
            ttk.Label(info_frame, text="Memory Usage:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Label(info_frame, text=mem_text).grid(row=2, column=1, sticky=tk.W, pady=5)
        except:
            pass
        
        # Generation settings
        ttk.Label(info_frame, text="\nGeneration Settings:", font=("Arial", 11, "bold")).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        ttk.Label(info_frame, text="Max tokens: 512").grid(row=4, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(info_frame, text="Temperature: 0.7").grid(row=5, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(info_frame, text="Top-p: 0.95").grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        # Intel optimizations
        ttk.Label(info_frame, text="\nIntel Optimizations:", font=("Arial", 11, "bold")).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        optimizations = []
        try:
            import intel_extension_for_pytorch
            optimizations.append("✓ Intel Extension for PyTorch")
        except:
            optimizations.append("✗ Intel Extension for PyTorch")
        
        try:
            import openvino
            optimizations.append("✓ OpenVINO Runtime")
        except:
            optimizations.append("✗ OpenVINO Runtime")
        
        for i, opt in enumerate(optimizations):
            ttk.Label(info_frame, text=opt).grid(row=8+i, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Button(settings_window, text="Close", command=settings_window.destroy).pack(pady=10)
    
    def show_intel_info(self):
        """Show Intel NPU information"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Intel NPU Information")
        info_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info = "Intel Hardware Information\n" + "="*50 + "\n\n"
        
        # CPU Info
        detector = IntelHardwareDetector()
        info += f"Intel CPU Detected: {detector.detect_intel_cpu()}\n"
        info += f"Processor: {platform.processor()}\n\n"
        
        # OpenVINO Devices
        try:
            from openvino.runtime import Core
            core = Core()
            info += "OpenVINO Available Devices:\n"
            for device in core.available_devices:
                info += f"  - {device}\n"
                try:
                    device_info = core.get_property(device, "FULL_DEVICE_NAME")
                    info += f"    Name: {device_info}\n"
                except:
                    pass
        except Exception as e:
            info += f"OpenVINO not available: {e}\n"
        
        info += "\n" + "="*50 + "\n"
        info += "NPU Optimization Tips:\n"
        info += "- Use INT8 quantization for best NPU performance\n"
        info += "- Keep batch size = 1 for optimal latency\n"
        info += "- Use static shapes when possible\n"
        info += "- Disable beam search for faster inference\n"
        
        text.insert("1.0", info)
        text.config(state=tk.DISABLED)
        
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)
    
    def clear_cache(self):
        """Clear model cache"""
        if messagebox.askyesno("Clear Cache", "This will clear the model cache. Continue?"):
            try:
                import shutil
                cache_dirs = [
                    Path.home() / ".cache" / "huggingface",
                    Path.home() / ".cache" / "torch",
                    Path.home() / ".cache" / "openvino",
                ]
                
                cleared = []
                for cache_dir in cache_dirs:
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                        cleared.append(cache_dir.name)
                
                if cleared:
                    messagebox.showinfo("Success", f"Cleared caches: {', '.join(cleared)}")
                else:
                    messagebox.showinfo("Info", "No caches found to clear")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache: {e}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """CATSEEK R2
Intel NPU Optimized Edition
Version 1.0
        
An AI-powered chat application using DeepSeek-R1-Zero
optimized for Intel Neural Processing Units (NPU).
        
Built with:
• Python 3.13
• DeepSeek-R1-Zero Model
• Intel Extension for PyTorch
• OpenVINO Runtime
• Transformers by Hugging Face
• Tkinter UI
        
© 2025 CATSEEK R2"""
        
        messagebox.showinfo("About CATSEEK R2", about_text)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts = """Keyboard Shortcuts:
        
Enter - Send message
Ctrl+Enter - New line in message
Ctrl+N - New chat
Ctrl+S - Save chat
Ctrl+O - Open chat
Ctrl+Q - Quit"""
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# ─── Main Entry Point ─────────────────────
def main():
    print("🚀 Starting CATSEEK R2 - Intel NPU Optimized Edition...")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if we need to install dependencies
    loading = LoadingScreen()
    loading.update_status("Checking dependencies...")
    
    installer = DependencyInstaller()
    
    # Check for missing packages
    missing = []
    for pkg in installer.required_packages:
        if not installer.check_package(pkg):
            missing.append(pkg)
    
    if missing:
        loading.update_status(f"Installing {len(missing)} missing packages...")
        
        def progress_callback(message, progress):
            loading.update_status(message, progress)
        
        if not installer.install_packages(progress_callback):
            loading.close()
            messagebox.showerror(
                "Dependency Error",
                "Failed to install some packages.\n"
                "Intel-specific packages may require manual installation.\n"
                "Core packages required:\n"
                "pip install torch transformers accelerate pillow"
            )
            # Continue anyway - Intel packages are optional
    
    loading.update_status("Starting CATSEEK R2...", 100)
    time.sleep(0.5)
    loading.close()
    
    # Import torch to check CUDA/Intel
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check for Intel Extension
        try:
            import intel_extension_for_pytorch as ipex
            print(f"Intel Extension for PyTorch: {ipex.__version__}")
        except:
            print("Intel Extension for PyTorch: Not installed")
        
        # Check for OpenVINO
        try:
            import openvino
            print(f"OpenVINO: {openvino.__version__}")
        except:
            print("OpenVINO: Not installed")
            
    except:
        pass
    
    # Start main application
    app = CATSEEKr2()
    app.run()

if __name__ == "__main__":
    main()
