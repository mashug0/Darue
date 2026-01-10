from main import NarrativeConsistencyAuditor
auditor = NarrativeConsistencyAuditor()
auditor.load_and_index_novel('Dataset/Books/The Count of Monte Cristo.txt')    
auditor.train_arbitrator('Dataset/train.csv', save_model_path='model.pkl')  