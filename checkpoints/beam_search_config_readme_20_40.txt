Beam Search Agent Configuration
==============================

Beam Width: 20
Search Depth: 40
Early Game Threshold: 512
Mid Game Threshold: 1024

Saved at: checkpoints/BeamSearchAgent_best_model_tile_1024.pth

This configuration achieved good results in training.
To recreate this agent, use:
agent = BeamSearchAgent(beam_width=20, search_depth=40)