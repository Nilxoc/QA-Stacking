import qaUtils
import dwave_networkx as dnx
from dwave_networkx.drawing.pegasus_layout import draw_pegasus_embedding
import sys
import matplotlib.pyplot as plt

ss = qaUtils.loadSampleset(sys.argv[1])
emb = ss.info['embedding_context']['embedding']
problemGr = ss.info['bqm'].to_networkx_graph()
draw_pegasus_embedding(dnx.pegasus_graph(16), emb=emb, embedded_graph=problemGr, unused_color=None, show_labels=True,crosses=True, verticalalignment='bottom')
plt.show()
