import requests
import time

API_URL = "http://localhost:5000/ask"

# Test various types of questions
test_questions = [
    # Greetings
    ("Hi", "greeting"),
    ("Good morning", "greeting"),
    ("How are you?", "greeting"),
    
    # Cricket questions (from your training data)
    ("Explain the basic rules of cricket.", "cricket"),
    ("What is an over in cricket?", "cricket"),
    ("What is LBW in cricket?", "cricket"),
    ("What is a century in cricket?", "cricket"),
    ("Who is the wicket-keeper?", "cricket"),
    
    # Non-cricket questions (should be declined)
    ("What is the capital of France?", "non-cricket"),
    ("How do I bake a cake?", "non-cricket"),
    ("Tell me about Python programming", "non-cricket"),
    
    # Farewell
    ("Thank you!", "greeting"),
    ("Bye", "greeting"),
]

def test_api():
    print("=" * 80)
    print("Testing Cricket Expert Chatbot API")
    print("=" * 80)
    
    for i, (question, category) in enumerate(test_questions, 1):
        print(f"\n[Question {i}] ({category}): {question}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                API_URL,
                json={"question": question, "max_length": 100},
                timeout=120  # Increased timeout to 2 minutes
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"[Answer]: {data['answer']}")
                print(f"[Time]: {elapsed_time:.2f}s")
            else:
                print(f"[Error]: HTTP {response.status_code}")
                print(f"[Response]: {response.text}")
        
        except requests.exceptions.ConnectionError:
            print("[Error]: Cannot connect. Is Flask running?")
            break
        except requests.exceptions.Timeout:
            print("[Error]: Request timed out. Model is taking too long to respond.")
        except Exception as e:
            print(f"[Error]: {str(e)}")
        
        print("-" * 80)
        time.sleep(0.5)  # Small delay between requests

if __name__ == "__main__":
    # First check if server is online
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        print("✓ Server is online!")
        print(f"Server info: {response.json()}\n")
    except:
        print("✗ Cannot connect to server. Please start Flask first!")
        exit(1)
    
    test_api()