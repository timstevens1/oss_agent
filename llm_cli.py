                                                                                                                                                                                            
# ------------------------------------------------------------------                                                                                                                           
#  llm_cli.py                                                                                                                                                                                  
# ------------------------------------------------------------------                                                                                                                           
                                                                                                                                                                                            
import os, sys                                                                                                                                                                                 
import json                                                                                                                                                                                    
import readline                                                                                                                                                                                
                                                                                                                                                                                            
# ------------------ GLOBAL MESSAGE STORE ------------------                                                                                                                                   
# This list represents the persistent client‑agent message cache.                                                                                                                              
# In a real application you would likely use a database or                                                                                                                                     
# knowledge‑graph instead of an in‑memory list.                                                                                                                                                
MESSAGE_STORE = []                                                                                                                                                                             
                                                                                                                                                                                            
def add_to_store(msg: str) -> None:                                                                                                                                                            
    """Store a user‑/agent message into the persistent cache."""                                                                                                                               
    MESSAGE_STORE.append(msg)                                                                                                                                                                  
                                                                                                                                                                                            
def clear_message_history() -> None:                                                                                                                                                           
    """Delete everything that is kept in the message store."""                                                                                                                                 
    global MESSAGE_STORE                                                                                                                                                                       
    MESSAGE_STORE.clear()                                                                                                                                                                      
    print("All messages have been cleared from the cache.") 
                                                                                                                                 
def smooth_exit():
    raise KeyboardInterrupt

# ------------------ SPECIAL COMMAND ROUTER ------------------                                                                                                                                 
def dispatch_special(command: str) -> None:                                                                                                                                                    
    """Map a /command to a specific function."""                                                                                                                                               
    routes = {                                                                                                                                                                                 
        "/clear_history": clear_message_history,                                                                                                                                               
        "/quit": lambda: smooth_exit(),                                                                                                                                                          
        # Other commands can be added here.                                                                                                                                                    
    }                                                                                                                                                                                          
    cmd = command.strip()                                                                                                                                                                      
    if cmd in routes:                                                                                                                                                                          
        routes[cmd]()                                                                                                                                                                          
    else:                                                                                                                                                                                      
        raise RuntimeError(f"Unknown special command: {command!r}")                                                                                                                            
                                                                                                                                                                                            
# ------------------ SIMPLE REPL LOOP ------------------                                                                                                                                       
def repl_loop() -> None:                                                                                                                                                                       
    """Top‑level REPL that supports arrow history & special commands."""                                                                                                                       
    # Initialise history handling                                                                                                                                                              
    hist_file = os.path.expanduser("~/.llm_cli_history")                                                                                                                                       
    if os.path.exists(hist_file):                                                                                                                                                              
        readline.read_history_file(hist_file)                                                                                                                                                  
    else:                                                                                                                                                                                      
        readline.write_history_file(hist_file)                                                                                                                                                         
                                                                                                                                                                                            
    while True:                                                                                                                                                                                
        try:                                                                                                                                                                                   
            user_input = input(">>> ")                                                                                                                                                         
            if not user_input:                                                                                                                                                                 
                continue                                                                                                                                                                       
                                                                                                                                                                                            
            # Store in the builtin revision list                                                                                                                                               
            readline.add_history_item(user_input)                                                                                                                                              
                                                                                                                                                                                            
            # Handle special commands                                                                                                                                                          
            if user_input.startswith("/"):                                                                                                                                                     
                dispatch_special(user_input)                                                                                                                                                   
                continue                                                                                                                                                                       
                                                                                                                                                                                            
            # <--- INSERT YOUR AGENT LOGIC HERE --->                                                                                                                                           
            # For demo we simply echo the input back.                                                                                                                                          
            response = f"Received: {user_input}"                                                                                                                                               
            print(response)                                                                                                                                                                    
                                                                                                                                                                                            
            # Save the message in the store                                                                                                                                                    
            add_to_store(user_input)                                                                                                                                                           
                                                                                                                                                                                            
        except KeyboardInterrupt:                                                                                                                                                              
            print("\nBye! Welcome again.")                                                                                                                                                     
            break                                                                                                                                                                              
        except Exception as exc:                                                                                                                                                               
            print(f"Oops {exc!r}")                                                                                                                                                             
                                                                                                                                                                                            
# ------------------ MAIN ENTRY POINT ------------------                                                                                                                                       
def main() -> None:                                                                                                                                                                            
    repl_loop()                                                                                                                                                                                
                                                                                                                                                                                            
if __name__ == "__main__":                                                                                                                                                                     
    main()