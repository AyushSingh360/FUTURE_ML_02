import pandas as pd
import numpy as np
import random
import os

def generate_dataset(num_samples=2000, output_path='../data/raw_tickets.csv'):
    categories = ['Billing Issue', 'Technical Problem', 'Account Access', 'Refund Request', 'General Inquiry']
    priorities = ['High', 'Medium', 'Low']

    templates = {
        'Billing Issue': [
            "Charged twice for my {} subscription.",
            "My payment failed but money was deducted for {}.",
            "Invoice {} shows an incorrect amount.",
            "How do I update my billing information for {}?",
            "Unrecognized charge of {} on my credit card.",
            "Can I get a receipt for my {} purchase?",
            "My {} payment is stuck in processing."
        ],
        'Technical Problem': [
            "Application crashes when uploading {} files.",
            "Error code occurs when I try to save {}.",
            "The {} feature is not working since the last update.",
            "System is running very slowly when processing {}.",
            "Cannot connect to the {} server.",
            "Data is missing from the {} dashboard.",
            "{} synchronization is failing repeatedly."
        ],
        'Account Access': [
            "Unable to login to my account after password reset.",
            "My account has been locked, please help me with {}.",
            "I forgot my {} password and my email is inaccessible.",
            "Need to change the admin email for account {}.",
            "Two-factor authentication code not arriving for {}.",
            "My {} profile is showing the wrong information.",
            "Cannot access the {} portal."
        ],
        'Refund Request': [
            "I want a refund for {}, it does not work as expected.",
            "Requesting money back for accidental purchase of {}.",
            "Cancel my subscription and refund the last {} charge.",
            "Not satisfied with {}, please process a refund.",
            "I was billed after cancellation, need a refund for {}.",
            "The {} I received is defective, refund please.",
            "I bought {} by mistake, can I get my money back?"
        ],
        'General Inquiry': [
            "How do I use the {} feature?",
            "Do you offer enterprise pricing for {}?",
            "Where can I find documentation for {}?",
            "Can I upgrade my plan to include {}?",
            "What are your support hours for {}?",
            "Is there a tutorial for {}?",
            "Does {} support integration with third-party apps?"
        ]
    }

    fillers = ['premium', 'monthly', 'pdf', 'image', 'API', 'dashboard', 'database', 'report', 'service', 'product', 'pro', 'starter', 'account', 'system']

    def generate_ticket(category):
        template = random.choice(templates[category])
        filler = random.choice(fillers)
        if "{}" in template:
            return template.format(filler)
        return template

    def assign_priority(category):
        if category in ['Technical Problem', 'Account Access']:
            return np.random.choice(priorities, p=[0.6, 0.3, 0.1])
        elif category == 'Refund Request':
            return np.random.choice(priorities, p=[0.4, 0.4, 0.2])
        elif category == 'Billing Issue':
            return np.random.choice(priorities, p=[0.5, 0.4, 0.1])
        else:  # General Inquiry
            return np.random.choice(priorities, p=[0.1, 0.4, 0.5])

    data = []
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    for i in range(num_samples):
        category = random.choice(categories)
        text = generate_ticket(category)
        
        # Add some noise to make the text slightly more realistic
        if random.random() > 0.8:
            text = text.lower() # someone forgot to capitalize
        if random.random() > 0.9:
            text = text + " Pls help fast." # urgency indicator
            
        priority = assign_priority(category)
        data.append({
            'ticket_id': f"TCK-{10000 + i}",
            'ticket_text': text,
            'category': category,
            'priority': priority
        })
        
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {len(df)} samples.")

if __name__ == "__main__":
    generate_dataset(2000, output_path='../data/raw_tickets.csv')
