import torch_geometric
from torch_geometric.utils import to_dense_adj

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score

# Train the given model on the given graph for num_epochs
def train(model, data, num_epochs, use_edge_index=False, learning_rate=0.01):
    if not use_edge_index:
        # Create the adjacency matrix
        # Important: add self-edges so a node depends on itself
        adj = to_dense_adj(data.edge_index)[0]
        adj += torch.eye(adj.shape[0])
    else:
        # Directly use edge_index
        adj = data.edge_index

    # Set up the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # A utility function to compute the f1 score
    def f1(outs, y, mask):
        return f1_score(outs[mask].argmax(dim=1),y[mask],average="micro")

    best_acc_val = -1
    for epoch in range(num_epochs):
        # Zero grads -> forward pass -> compute loss -> backprop
        optimizer.zero_grad()
        outs = model(data.x, adj)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Compute scores, print only if this is the best result so far
        acc_val = f1(outs, data.y, data.val_mask)
        acc_test = f1(outs, data.y, data.test_mask)
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')

# Train the given model on a series of graphs
def train_ppi(model, data_list, num_epochs, use_edge_index=False, learning_rate=0.01):
    adj_list = []
    for data in data_list:
        if not use_edge_index:
            # Create the adjacency matrix
            # Important: add self-edges so a node depends on itself
            adj = to_dense_adj(data.edge_index)[0]
            adj += torch.eye(adj.shape[0])
        else:
            # Directly use edge_index
            adj = data.edge_index

        adj_list.append(adj)

    # Set up the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # A utility function to compute the f1 score
    def f1(outs, y):
        return f1_score(outs.argmax(dim=1),y.argmax(dim=1),average="micro")

    for epoch in range(num_epochs):
        for graph_number, (data, adj) in enumerate(zip(data_list[:-2],adj_list[:-2])):
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            outs = model(data.x, adj)
            loss = loss_fn(outs, data.y.argmax(dim=1))
            loss.backward()
            optimizer.step()

            # print score after every graph
            outs1 = model(data_list[-2].x, adj_list[-2])
            outs2 = model(data_list[-1].x, adj_list[-1])
            acc_val = f1(outs1, data_list[-2].y)
            acc_test = f1(outs2, data_list[-1].y)
            print(f'[Graph {graph_number+1}/{len(data_list)-2}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')