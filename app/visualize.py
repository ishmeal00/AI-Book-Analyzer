# app/visualize.py
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def build_knowledge_graph(keywords, entities=None, save_path="output/knowledge_graph.png"):
    G = nx.Graph()
    for kw in keywords:
        G.add_node(kw, type="kw")
    if entities:
        for ent, label in entities:
            G.add_node(ent, type=label)
            if keywords:
                G.add_edge(ent, keywords[0])
    for i in range(len(keywords)-1):
        G.add_edge(keywords[i], keywords[i+1])
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1800, font_size=10)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_wordcloud(text, save_path="output/wordcloud.png"):
    wc = WordCloud(width=1400, height=700, background_color="white").generate(text)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(14,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
