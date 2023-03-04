import dwave_networkx as dnx
from dwave_networkx.drawing.pegasus_layout import draw_pegasus_embedding

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 

from stackingPallet import PalletQUBOGenerator
import minorminer
import seaborn as sns

import qaUtils

def generateColorPallete(bqm):
  palette = sns.color_palette('colorblind', len(bqm.variables))
  res = {}
  for i, var in enumerate(bqm.variables):
    color = palette[i]
    res[var] = (color[0],color[1],color[2],1)
  return res

if __name__ == '__main__':
  gen = PalletQUBOGenerator([[0,1],[1,0]])
  palette = generateColorPallete(gen.bqm)

  sGraph = gen.bqm.to_networkx_graph()
  tGraph = dnx.pegasus_graph(2)
  emb = minorminer.find_embedding(sGraph, tGraph)
  draw_pegasus_embedding(tGraph, emb, chain_color = palette, unused_color=(.5,.5,.5,.5),node_size=100, crosses=True)
    
  artists = [mlines.Line2D([], [], color=palette[var],marker='o', linestyle=None, markersize=10, label=var) for var in palette]
  labels = gen.bqm.variables
  mathLabels = []
  for label in labels:
    #Matplotlib use \wedge instead of \land for the boolean and symbol, but is mostly similar to LaTeX math
    mathLabels.append(qaUtils.varNameToLatex(label).replace('\\land','\\wedge'))
  
  plt.legend(artists,mathLabels)
  plt.show()
