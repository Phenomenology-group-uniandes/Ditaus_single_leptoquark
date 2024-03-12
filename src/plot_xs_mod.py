import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT

from xs_scan import calculate_cross_section

ufo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "LO_LQ_S1~")

mg5_bin_path = os.path.join(os.sep, "Collider", "MG5_aMC_v3_1_0", "bin", "mg5_aMC")

outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(outputs_dir, exist_ok=True)
plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdfs")
n_cores = mp.cpu_count()
n_workers = n_cores - 2 if n_cores > 2 else 1

n_events = int(1e5) 
#number of events 

print("Loading UFO model from:\n", ufo_path)
print("Using MG5 binary from:\n", mg5_bin_path)
print("Saving MG5 outputs in directory:\n", outputs_dir)
print("Using", n_workers, "workers")
print("Using", n_events, "events per run")

#Acopling params
params_dict = {
    # mg5_name: value
    "yrr1x1": 1.000000e-01,
    "yrr1x2": 0.000000e00,
    "yrr1x3": 0.000000e00,
    "yrr2x1": 0.000000e00,
    "yrr2x2": 1.000000e-01,
    "yrr2x3": 0.000000e00,
    "yrr3x1": 0.000000e00,
    "yrr3x2": 0.000000e00,
    "yrr3x3": 1.000000e-01
}

# Set the mass range for the scan
m_min = 0.5
m_max = 3.5
step = 0.25
masses = np.arange(m_min, m_max + step, step)

# Declare the list to store the used seeds
seeds = []

def get_xs(mass: float):
    xs = calculate_cross_section(
        mass,
        params_dict,
        ufo_path,
        mg5_bin_path,
        outputs_dir,
        seeds,
        n_events,
        n_workers,
    )
    return (mass, xs)

results = list(map(get_xs, masses))

df = pd.DataFrame(results, columns=["mass", "xs"])

# Draw the results using ROOT
canva = ROOT.TCanvas("canva", "canva", 800, 600)
x, y = zip(*results)
gr = ROOT.TGraph(len(x), np.array(x), np.array(y))
gr.SetTitle("Cross section scan;Mass [TeV];Cross section [pb]")
gr.SetMarkerStyle(20)
gr.Draw("ALP")
legend = ROOT.TLegend(0.5, 0.7, 0.88, 0.9)
legend.AddEntry(gr, "non-resonant ditau production", "lp")
legend.Draw()  # Draw the legend
canva.SetLogy(1)  # Set y-axis to logarithmic scale
canva.Draw()



