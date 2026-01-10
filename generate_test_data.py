"""
Test Data Generator for Narrative Consistency Auditor
Creates test.csv with character backstory claims for evaluation.
"""
import pandas as pd
import os


def generate_test_data():
    """Generate test dataset with character backstories from The Count of Monte Cristo and In Search of the Castaways."""
    
    # Test cases for narrative consistency
    test_data = [
        # The Count of Monte Cristo - CONSISTENT claims
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'was a sailor who became a ship captain',
            'label': 1
        },
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'was imprisoned in the Château d\'If',
            'label': 1
        },
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'was betrayed by Fernand and Danglars',
            'label': 1
        },
        {
            'character_name': 'Mercedes',
            'backstory_claim': 'was engaged to Edmond Dantès',
            'label': 1
        },
        {
            'character_name': 'Abbé Faria',
            'backstory_claim': 'was a prisoner who befriended Edmond',
            'label': 1
        },
        {
            'character_name': 'Fernand Mondego',
            'backstory_claim': 'was jealous of Edmond Dantès',
            'label': 1
        },
        
        # The Count of Monte Cristo - CONTRADICTORY claims
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'was born in Paris and never sailed the seas',
            'label': 0
        },
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'was a wealthy nobleman from birth',
            'label': 0
        },
        {
            'character_name': 'Mercedes',
            'backstory_claim': 'was married to Albert before meeting Edmond',
            'label': 0
        },
        {
            'character_name': 'Abbé Faria',
            'backstory_claim': 'was a guard at the Château d\'If',
            'label': 0
        },
        
        # In Search of the Castaways - CONSISTENT claims
        {
            'character_name': 'Captain Grant',
            'backstory_claim': 'was a Scottish sea captain who was shipwrecked',
            'label': 1
        },
        {
            'character_name': 'Lord Glenarvan',
            'backstory_claim': 'owned the yacht Duncan',
            'label': 1
        },
        {
            'character_name': 'Mary Grant',
            'backstory_claim': 'was Captain Grant\'s daughter',
            'label': 1
        },
        {
            'character_name': 'Paganel',
            'backstory_claim': 'was a French geographer who joined the expedition',
            'label': 1
        },
        
        # In Search of the Castaways - CONTRADICTORY claims
        {
            'character_name': 'Captain Grant',
            'backstory_claim': 'was found immediately after the shipwreck',
            'label': 0
        },
        {
            'character_name': 'Lord Glenarvan',
            'backstory_claim': 'was a poor fisherman with no resources',
            'label': 0
        },
        {
            'character_name': 'Paganel',
            'backstory_claim': 'was an English military officer',
            'label': 0
        },
        
        # Mixed - edge cases
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'changed his identity after escaping prison',
            'label': 1
        },
        {
            'character_name': 'Edmond Dantès',
            'backstory_claim': 'discovered a hidden treasure',
            'label': 1
        },
        {
            'character_name': 'Mercedes',
            'backstory_claim': 'eventually married Fernand Mondego',
            'label': 1
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Save to CSV
    output_path = 'Dataset/test.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Test data generated successfully!")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(f"  Consistent (1):     {(df['label'] == 1).sum()}")
    print(f"  Contradictory (0):  {(df['label'] == 0).sum()}")
    
    # Show sample data
    print(f"\nSample data (first 5 rows):")
    print(df.head().to_string(index=False))
    
    return df


def generate_training_data():
    """Generate train.csv with more examples for ML model training."""
    
    training_data = [
        # The Count of Monte Cristo - Expanded training set
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'was first mate of the Pharaon', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'was about to marry Mercedes', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'was accused of being a Bonapartist', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'spent years in solitary confinement', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'learned many skills from Abbé Faria', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'escaped by taking the place of a dead body', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'became the Count of Monte Cristo', 'label': 1},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'sought revenge against his enemies', 'label': 1},
        
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'was a doctor in Paris', 'label': 0},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'never went to sea', 'label': 0},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'was imprisoned for theft', 'label': 0},
        {'character_name': 'Edmond Dantès', 'backstory_claim': 'died in the Château d\'If', 'label': 0},
        
        {'character_name': 'Mercedes', 'backstory_claim': 'was a Catalan woman from the village', 'label': 1},
        {'character_name': 'Mercedes', 'backstory_claim': 'waited for Edmond initially', 'label': 1},
        {'character_name': 'Mercedes', 'backstory_claim': 'married Fernand after believing Edmond was dead', 'label': 1},
        {'character_name': 'Mercedes', 'backstory_claim': 'had a son named Albert', 'label': 1},
        
        {'character_name': 'Mercedes', 'backstory_claim': 'was born in Paris', 'label': 0},
        {'character_name': 'Mercedes', 'backstory_claim': 'never knew Edmond Dantès', 'label': 0},
        
        {'character_name': 'Fernand Mondego', 'backstory_claim': 'was a fisherman who loved Mercedes', 'label': 1},
        {'character_name': 'Fernand Mondego', 'backstory_claim': 'conspired to have Edmond imprisoned', 'label': 1},
        {'character_name': 'Fernand Mondego', 'backstory_claim': 'became Count de Morcerf', 'label': 1},
        {'character_name': 'Fernand Mondego', 'backstory_claim': 'was a loyal friend to Edmond', 'label': 0},
        
        {'character_name': 'Abbé Faria', 'backstory_claim': 'was an Italian priest and scholar', 'label': 1},
        {'character_name': 'Abbé Faria', 'backstory_claim': 'knew about a hidden treasure', 'label': 1},
        {'character_name': 'Abbé Faria', 'backstory_claim': 'educated Edmond during imprisonment', 'label': 1},
        {'character_name': 'Abbé Faria', 'backstory_claim': 'died in prison', 'label': 1},
        {'character_name': 'Abbé Faria', 'backstory_claim': 'escaped with Edmond', 'label': 0},
        
        # In Search of the Castaways
        {'character_name': 'Captain Grant', 'backstory_claim': 'commanded the ship Britannia', 'label': 1},
        {'character_name': 'Captain Grant', 'backstory_claim': 'was searching for a location to found a Scottish colony', 'label': 1},
        {'character_name': 'Captain Grant', 'backstory_claim': 'sent a message in a bottle', 'label': 1},
        {'character_name': 'Captain Grant', 'backstory_claim': 'was never lost at sea', 'label': 0},
        
        {'character_name': 'Lord Glenarvan', 'backstory_claim': 'was a wealthy Scottish nobleman', 'label': 1},
        {'character_name': 'Lord Glenarvan', 'backstory_claim': 'led the rescue expedition', 'label': 1},
        {'character_name': 'Lord Glenarvan', 'backstory_claim': 'owned the Duncan yacht', 'label': 1},
        {'character_name': 'Lord Glenarvan', 'backstory_claim': 'was Captain Grant\'s enemy', 'label': 0},
        
        {'character_name': 'Mary Grant', 'backstory_claim': 'was searching for her father', 'label': 1},
        {'character_name': 'Mary Grant', 'backstory_claim': 'joined the expedition on the Duncan', 'label': 1},
        {'character_name': 'Mary Grant', 'backstory_claim': 'had a brother named Robert', 'label': 1},
        {'character_name': 'Mary Grant', 'backstory_claim': 'was Lord Glenarvan\'s sister', 'label': 0},
        
        {'character_name': 'Paganel', 'backstory_claim': 'was a member of the Geographical Society', 'label': 1},
        {'character_name': 'Paganel', 'backstory_claim': 'boarded the Duncan by mistake', 'label': 1},
        {'character_name': 'Paganel', 'backstory_claim': 'was known for his absent-mindedness', 'label': 1},
        {'character_name': 'Paganel', 'backstory_claim': 'was the ship\'s captain', 'label': 0},
    ]
    
    df = pd.DataFrame(training_data)
    
    # Save to CSV
    output_path = 'Dataset/train.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n{'='*60}")
    print(f"✓ Training data generated successfully!")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(f"  Consistent (1):     {(df['label'] == 1).sum()}")
    print(f"  Contradictory (0):  {(df['label'] == 0).sum()}")
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GENERATING TEST AND TRAINING DATA")
    print("="*60 + "\n")
    
    # Create Dataset directory if it doesn't exist
    os.makedirs('Dataset', exist_ok=True)
    
    # Generate both datasets
    test_df = generate_test_data()
    train_df = generate_training_data()
    
    print("\n" + "="*60)
    print("  GENERATION COMPLETE!")
    print("="*60)
    print("\nNow you can:")
    print("1. Run: python main.py")
    print("2. Train model: auditor.train_arbitrator('Dataset/train.csv')")
    print("3. Test system with Dataset/test.csv")
    print("="*60 + "\n")
