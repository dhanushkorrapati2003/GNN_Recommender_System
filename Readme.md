#T-GNNs

This project involves implementing a Temporal Graph Neural Network (GNN) for recommender systems. The code includes:

1. **GraphRec Model**: A neural network designed to predict user-item interactions using embeddings for users, items, and ratings. It incorporates both the user's interaction history and social connections to model complex patterns in user behavior.
   
2. **UserModel and ItemModel**: Sub-models that handle the aggregation of user-item interactions and social connections. They utilize attention mechanisms to weigh the importance of different interactions and neighbors.

3. **Data Preparation**: A `GraphDataset` class is provided to organize the input data for the model. It manages users’ interaction history and social connections, truncating long sequences and handling padding where necessary.

4. **Training Support**: The `collate_fn` function handles batch preparation for training, padding the data sequences and ensuring they are of equal lengths to be processed by the model.

This is temporary code designed to explore temporal GNNs for recommendation systems, with a focus on understanding how temporal relationships between user interactions can optimize recommendations.