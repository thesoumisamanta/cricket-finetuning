import requests
import time
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

API_URL = "http://localhost:5000/ask"

def print_bot_response(message):
    """Print bot response in green"""
    print(f"{Fore.GREEN}🏏 Cricket Bot: {message}{Style.RESET_ALL}")

def print_user_message(message):
    """Print user message in blue"""
    print(f"{Fore.BLUE}👤 You: {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error in red"""
    print(f"{Fore.RED}❌ Error: {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info in yellow"""
    print(f"{Fore.YELLOW}ℹ️  {message}{Style.RESET_ALL}")

def chat():
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}🏏 Welcome to Cricket Expert Chatbot! 🏏{Style.RESET_ALL}")
    print("=" * 80)
    print_info("I can answer questions about cricket rules, terms, and general cricket knowledge.")
    print_info("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print_info("✓ Connected to Cricket Bot API\n")
    except:
        print_error("Cannot connect to server. Please start Flask with: python3 app.py")
        return
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.BLUE}👤 You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print_bot_response("Goodbye! Thanks for chatting about cricket. Come back anytime! 🏏")
                break
            
            # Send request to API
            start_time = time.time()
            print_info("Thinking...")
            
            response = requests.post(
                API_URL,
                json={"question": user_input, "max_length": 150},
                timeout=120
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data['answer']
                
                # Print bot response
                print(f"{Fore.GREEN}🏏 Cricket Bot: {answer}{Style.RESET_ALL}")
                print_info(f"Response time: {elapsed_time:.2f}s\n")
                
                # Store in history
                conversation_history.append({
                    "user": user_input,
                    "bot": answer,
                    "time": elapsed_time
                })
            else:
                print_error(f"HTTP {response.status_code}: {response.text}\n")
        
        except requests.exceptions.Timeout:
            print_error("Request timed out. The model is taking too long. Try a shorter question.\n")
        except requests.exceptions.ConnectionError:
            print_error("Lost connection to server. Please restart Flask.\n")
            break
        except KeyboardInterrupt:
            print("\n")
            print_bot_response("Chat interrupted. Goodbye! 🏏")
            break
        except Exception as e:
            print_error(f"An error occurred: {str(e)}\n")
    
    # Print conversation summary
    if conversation_history:
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}📊 Conversation Summary{Style.RESET_ALL}")
        print("=" * 80)
        print(f"Total questions asked: {len(conversation_history)}")
        avg_time = sum(conv['time'] for conv in conversation_history) / len(conversation_history)
        print(f"Average response time: {avg_time:.2f}s")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 🏏")