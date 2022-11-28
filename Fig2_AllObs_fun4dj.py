import numpy as np
import ROOT
import pickle #import 2.76 data

import scipy
from scipy import interpolate

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
pd.options.plotting.backend = "plotly"

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

df_cols = ['ObsType','Observables', 'Centrality' , 'Correlation',"stat_err","year"]
df_new = pd.DataFrame(columns=df_cols)

#SPC
for i in range(0,len(obsTypeStr_SPC)):
	gr = f_SPC.Get("{:s}{:s}".format(obsTypeStr_SPC[i],"_Stat"));
	x,y,_,yerr = JPyPlotRatio.TGraphErrorsToNumpy(gr);
	for j in range(0,len(x)):
		df_new = df_new.append({'ObsType': "$\\langle\\cos\\left(a_{1} n_1 \\Psi_{n_1}+\\cdots+ a_{k} n_k \\Psi_{n_k}\\right) \\rangle_{GE}$",
			'Observables': plabel_SPC[i], 'Centrality': x[j], 'Correlation': np.absolute(y[j]),"stat_err":yerr[j],"year":2022}, ignore_index=True) # yuck


# SC(k,l,m)
for i in range(0,len(obsTypeStr_HSC)):
	gr = f_HSC.Get("graph_{:s}ALICE".format(obsTypeStr_HSC[i]));
	x,y,_,yerr = JPyPlotRatio.TGraphErrorsToNumpy(gr);
	for j in range(0,len(x)):
		df_new = df_new.append({'ObsType': "$SC(k,l,m)=\\left<v_k^2v_l^2v_m^2\\right>_c$",
		'Observables': plabel_HSC[i], 'Centrality': x[j], 'Correlation': np.absolute(y[j]),"stat_err":yerr[j],"year":2021}, ignore_index=True) # yuck

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
			df_new = df_new.append({'ObsType': "$SC(n,m)=\\left<v_n^2v_m^2\\right>_c$",
			'Observables': label, 'Centrality': scpubd[s][0][j], 'Correlation': np.absolute(scpubd[s][1][j]),"stat_err":scpubd[s][2][j],"year":2016}, ignore_index=True) # yuck
			

#mymarkers = {"SPC": 's', "HSC": 'o', "NSC":'*'}
print(df_new)

plt = px.scatter(df_new, x="Centrality", y="Observables", 
				 color="ObsType", hover_data=['ObsType'],
				 size="Correlation", 
                 size_max=100)

plt.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.8),
    xaxis_title="Centrality Percentile",
    yaxis_title="",
    font=dict(
       size=16
    )
)

plt.show()

#plt.text(0.85,0.75,toptitle,fontsize=10);
#plt.title('Summary of Run1 Correlators')
#plt.savefig("figs/Fig2_allobs_fun4dj.pdf")
#plt.show()