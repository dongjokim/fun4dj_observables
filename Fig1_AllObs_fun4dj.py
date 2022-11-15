import numpy as np
import ROOT
import pickle #import 2.76 data

import scipy
from scipy import interpolate

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("JPyPlotRatio");

import JPyPlotRatio

#SPC
f_SPC = ROOT.TFile("data/ALICE_SPC_Run1.root","read");

obsTypeStr_SPC  = ["4Psi4_n4Psi2","6Psi3_n6Psi2","6Psi6_n6Psi2","6Psi6_n6Psi3",
		"2Psi2_3Psi3_n5Psi5","8Psi2_n3Psi3_n5Psi5","2Psi2_n6Psi3_4Psi4","2Psi2_4Psi4_n6Psi6",
		"2Psi2_n3Psi3_n4Psi4_5Psi5"
		];
plabel_SPC     = ["$\\langle cos[4(\\Psi_{4}-\\Psi_{2})]\\rangle$",
	      "$\\langle cos[6(\\Psi_{2}-\\Psi_{3})]\\rangle$",
	      "$\\langle cos[6(\\Psi_{6}-\\Psi_{2})]\\rangle$",
	      "$\\langle cos[6(\\Psi_{6}-\\Psi_{3})]\\rangle$", # 2 har
	      "$\\langle cos[2\\Psi_{2}+3\\Psi_{3}-5\\Psi_{5}]\\rangle$",
	      "$\\langle cos[8\\Psi_{2}-3\\Psi_{3}-5\\Psi_{5}]\\rangle$",
	      "$\\langle cos[2\\Psi_{2}-6\\Psi_{3}+4\\Psi_{4}]\\rangle$",
	      "$\\langle cos[2\\Psi_{2}+4\\Psi_{4}-6\\Psi_{6}]\\rangle$", # 3 har
	      "$\\langle cos[2\\Psi_{2}-3\\Psi_{3}-4\\Psi_{4}+5\\Psi_{5}]\\rangle$" # 4 har
	      ];

#NSC
f_HSC = ROOT.TFile("data/HighSC_20220407.root","read");
obsTypeStr_HSC  = ["NSC234","NSC235"];
plabel_HSC     = ["NSC(2,3,4)","NSC(2,3,5)"];
#load 2.76 TeV results from hepdata file
with open("data/sc2760_hepdata.pickle","rb") as fd:
	scpubd = pickle.load(fd);

def RemovePoints(arrays, pointIndices):
	return tuple([np.delete(a,pointIndices) for a in arrays]);


xlimits = [(-1.,52.)];
ylimits = [(-0.3,0.55)];

xtitle = ["Centrality percentile"];
ytitle = ["Correlations"];
ytitleRight = ["NSC(k,l,m)"];
# Following two must be added
toptitle = "PbPb $\\sqrt{s_{NN}}$ = 2.76 TeV"; # need to add on the top
dataDetail = "$0.2 < p_\\mathrm{T} < 5.0\\,\\mathrm{GeV}/c$\n$|\\eta| < 0.8$";
plables = [ "(a)", "(b)" ];



#plot.EnableLatex(True);

df = pd.DataFrame();
gr = f_SPC.Get("{:s}{:s}".format(obsTypeStr_SPC[0],"_Stat"));
x,y,_,yerr = JPyPlotRatio.TGraphErrorsToNumpy(gr);
df['Centrality'] = x.tolist()

df_cols = ['ObsType','Observables', 'Centrality' , 'Correlation',"stat_err"]
df_new = pd.DataFrame(columns=df_cols)

#SPC
for i in range(0,len(obsTypeStr_SPC)):
	gr = f_SPC.Get("{:s}{:s}".format(obsTypeStr_SPC[i],"_Stat"));
	x,y,_,yerr = JPyPlotRatio.TGraphErrorsToNumpy(gr);
	for j in range(0,len(x)):
		df_new = df_new.append({'ObsType': "$\\langle\\cos\\left(a_{1} n_1 \\Psi_{n_1}+\\cdots+ a_{k} n_k \\Psi_{n_k}\\right) \\rangle_{GE}$",'Observables': plabel_SPC[i], 'Centrality': x[j], 'Correlation': y[j],"stat_err":yerr[j]}, ignore_index=True) # yuck
	df[obsTypeStr_SPC[i]] = y.tolist()
	#g = sns.scatterplot(data=df, x="Centrality", y=obsTypeStr_SPC[i], size=obsTypeStr_SPC[i], legend=False, sizes=(20, 400))

# SC(k,l,m)
for i in range(0,len(obsTypeStr_HSC)):
	gr = f_HSC.Get("graph_{:s}ALICE".format(obsTypeStr_HSC[i]));
	x,y,_,yerr = JPyPlotRatio.TGraphErrorsToNumpy(gr);
	for j in range(0,len(x)):
		df_new = df_new.append({'ObsType': "SC$(k,l,m)=\\left<v_k^2v_l^2v_m^2\\right>_c$",'Observables': plabel_HSC[i], 'Centrality': x[j], 'Correlation': y[j],"stat_err":yerr[j]}, ignore_index=True) # yuck
#	df[obsTypeStr_HSC[i]] = y.tolist()
#	sns.scatterplot(data=df, x="Centrality", y=obsTypeStr_HSC[i], size=obsTypeStr_HSC[i], legend=False, sizes=(20, 400))

#SC(k,l)
for index,s in enumerate(list(scpubd)):
	obs = s[5:8];
	print(obs)
	label = "SC({},{})".format(*obs[0:2]);
	print(label)
	if obs[2] == 'N':
		label = "N"+label;
		#arrays = RemovePoints(scpubd[s],np.array([6,7]));
		print(scpubd[s][0],scpubd[s][1])
		for j in range(0,len(scpubd[s][0])):
			df_new = df_new.append({'ObsType': "SC$(n,m)=\\left<v_n^2v_m^2\\right>_c$",'Observables': label, 'Centrality': scpubd[s][0][j], 'Correlation': scpubd[s][1][j],"stat_err":scpubd[s][2][j]}, ignore_index=True) # yuck
			
# colors = {"SPC":'red', "HSC":'blue'}
# markers = {"SPC": 's', "HSC": 'o'}
# color_labels = df_new['Observables'].unique()
# print(color_labels)
# # List of colors in the color palettes
# rgb_values = sns.color_palette("Set2", 11)

# # Map continents to the colors
# color_map = dict(zip(color_labels, rgb_values))
#c=df_new['Observables'].map(color_map),
g = sns.scatterplot(data=df_new, x="Centrality", y="Correlation", size="Correlation", 
	hue='ObsType', sizes=(20, 400))
h,l = g.get_legend_handles_labels();
print(df_new)
plt.legend(h[1:4], l[1:4], bbox_to_anchor=(0., 1), loc='upper left', borderaxespad=0)
# Set x-axis label
plt.xlabel(xtitle[0],fontsize=15)
# Set y-axis label
plt.ylabel(ytitle[0],fontsize=15)
g.set(xlim=(-1,60),ylim=(-0.25,1.1))

# show the graph
plt.text(0.85,0.75,toptitle,fontsize=10);

plt.savefig("figs/allobs_fun4dj.pdf")
plt.show()



