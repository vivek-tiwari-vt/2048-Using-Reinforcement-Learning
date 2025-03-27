Beam Search Agent Configuration
==============================

Beam Width: 15
Search Depth: 30
Early Game Threshold: 512
Mid Game Threshold: 1024

Saved at: checkpoints/BeamSearchAgent_final_model.pth

This configuration achieved good results in training.
To recreate this agent, use:
agent = BeamSearchAgent(beam_width=15, search_depth=30)