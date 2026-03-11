import sys
sys.path.append('..')

if len(sys.argv) != 3:
    print("Usage: python3 trackLinking_pt_eta.py PT ETA")
    sys.exit()

import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import mplhep as hep
import vector as vec
import matplotlib
from tqdm import tqdm
import math

from Timing.plotting import *

from numba import prange, njit
import awkward.numba

plt.style.use(hep.style.CMS)

@njit
def flatten_numba(a):
    return [x[0] if len(x) else 0 for x in a]

def dist(refEta, refPhi, otherTsEta, otherTsPhi):
    return ((otherTsEta-refEta)**2 + (otherTsPhi-refPhi)**2)**0.5

@njit
def dist_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    out = []
    for i in range(len(otherTsEta)):
        distance = ((otherTsEta[i] - refEta) ** 2 + (otherTsPhi[i] - refPhi) ** 2) ** 0.5
        out.append(distance)
    return np.array(out)

@njit
def distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    out = []
    for i in range(len(otherTsEta)):
        deltaPhi = otherTsPhi[i] - refPhi
        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
        distance = ((otherTsEta[i] - refEta) ** 2 + deltaPhi ** 2) ** 0.5
        out.append(distance)
    return np.array(out)

def find_track_id(array, number):
    try:
        return np.where(array == number)[0][0]
    except:
        return -1

def load_branch_with_highest_cycle(file, branch_name):
    # Get all keys in the file
    all_keys = file.keys()
    # Filter keys that match the specified branch name
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    # Find the key with the highest cycle
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    # Load the branch with the highest cycle
    branch = file[highest_cycle_key]
    return branch

def distance(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5

# returns res that contains the parameters, the chi squared and
# the counts and bins used to plot the data
def gauss_fit(data, init_parms, bins=300):
    hist, nbins = np.histogram(data, bins=bins)
    nbins = 0.5 * (bins[1:] + bins[:-1])
    errors = [np.sqrt(oh+1) for oh in hist]
    init_parameters = init_parms
    cost_func = cost.LeastSquares(nbins, hist, errors, model)
    min_obj = Minuit(cost_func, *init_parameters)
    res = min_obj.migrad()
    chi2 = min_obj.fval/(len(nbins[:-1])-3)
    return res, chi2, hist, nbins[:-1]

#same as above but plots also the data
def gauss_fit_and_plot(data, init_parms, label="data", colors=["midnightblue","dodgerblue"], bins=300):
    res, chi2, hists, newbins = gauss_fit(data, init_parms, bins=bins)
    y = model(newbins, *res.values)
    plt.plot(newbins, y, label=f'gauss fit\n   $\sigma$ = {res.values[2]:.3f} $\pm$ {res.errors[2]:.3f}\n   $x_0$ = {res.values[1]:.3f} $\pm$ {res.errors[1]:.3f} \n   $\chi^2_0$ = {chi2:.3f}', color=colors[0], linewidth=2)
    plt.hist(np.array(data), bins=bins, color=colors[1], alpha=0.7)
    plt.legend(fontsize=16)
    plt.grid()
    return res, chi2

# quick plot with list, np array or flattened awkward array
def myhist(X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts / bin', color='dodgerblue', alpha=1, fill='stepfilled', range=None, label="data"):
    #plt.figure(dpi=100)
    if range==None:
        plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, label=label)
    else:
        plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, range=range, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def find_track_ids(array, numbers):
    # Create a dictionary mapping values to indices
    index_map = {val: idx for idx, val in enumerate(array)}
    # Use vectorized lookup, returning -1 if number is not found
    return np.array([index_map.get(num, -1) for num in numbers])

@njit
def find_in_array_with_none(array, match):
    for i, el in enumerate(array):
        if len(el) == 0: continue
        if el[0] == match:
            return i
    return -1

C_CM_PER_NS = 29.9792458

def sigmaEp(p, isElectron):
    a = np.where(isElectron, 0.25, 1.0)
    b = np.where(isElectron, 0.02, 0.10)
    f = np.where(isElectron, 1.0, 0.7)

    Eexp = f * p
    relSigma = np.sqrt((a*a)/Eexp + b*b)
    return Eexp * relSigma

##########
#  MAIN  #
##########

tracksKeys = [
 'track_id',
 'track_hgcal_x',
 'track_hgcal_y',
 'track_hgcal_z',
 'track_hgcal_eta',
 'track_hgcal_phi',
 'track_hgcal_pt',
 'track_pt',
 'track_p',
 'track_missing_outer_hits',
 'track_missing_inner_hits',
 'track_quality',
#  'track_charge',
#  'track_time',
#  'track_time_quality',
#  'track_time_err',
#  'track_beta',
 'track_time_mtd',
 'track_time_mtd_err',
#  'track_pos_mtd',
 'track_pos_mtd/track_pos_mtd.theVector.theX',
 'track_pos_mtd/track_pos_mtd.theVector.theY',
 'track_pos_mtd/track_pos_mtd.theVector.theZ',
 'track_nhits',
 'track_isMuon',
 'track_isTrackerMuon'
]

simTsKeys = [
 'regressed_energy',
 'raw_energy',
 'trackIdx',
 # 'raw_em_energy',
 # 'raw_pt',
 # 'raw_em_pt',
 # 'barycenter_x',
 # 'barycenter_y',
 'barycenter_z',
 'barycenter_eta',
 'barycenter_phi',
 #'trackTime',
 # 'EV1',
 # 'EV2',
 # 'EV3',
 # 'eVector0_x',
 # 'eVector0_y',
 # 'eVector0_z',
 # 'sigmaPCA1',
 # 'sigmaPCA2',
 # 'sigmaPCA3',
 # 'regressed_pt',
 'pdgID'
]

assKeys = [
#  'ticlTracksterLinks_recoToSim_CP',
#  'ticlTracksterLinks_recoToSim_CP_score',
#  'ticlTracksterLinks_recoToSim_CP_sharedE',
 'ticlTracksterLinks_simToReco_SC',
 'ticlTracksterLinks_simToReco_SC_score',
 'ticlTracksterLinks_simToReco_SC_sharedE',
#     'ticlCandidate_simToReco_CP_score',
#     'ticlCandidate_simToReco_CP_sharedE'
#  'ticlCandidate_recoToSim_SC',
#  'ticlCandidate_recoToSim_SC_score',
#  'ticlCandidate_recoToSim_SC_sharedE',
#  'ticlCandidate_simToReco_SC',
#  'ticlCandidate_simToReco_SC_score',
#  'ticlCandidate_simToReco_SC_sharedE',
          ]

tsKeys = [
#  'NTracksters',
#  'NClusters',
 'time',
 'timeError',
 'regressed_energy',
 'raw_energy',
 'raw_em_energy',
 'raw_pt',
 'raw_em_pt',
 'barycenter_x',
 'barycenter_y',
 'barycenter_z',
 'barycenter_eta',
 'barycenter_phi',
#  'EV1',
#  'EV2',
#  'EV3',
#  'eVector0_x',
#  'eVector0_y',
#  'eVector0_z',
#  'sigmaPCA1',
#  'sigmaPCA2',
#  'sigmaPCA3',
#  'id_probabilities',
#  'vertices_indexes',
#  'vertices_x',
#  'vertices_y',
#  'vertices_z',
#  'vertices_time',
#  'vertices_timeErr',
#  'vertices_energy',
#  'vertices_correctedEnergy',
#  'vertices_correctedEnergyUncertainty',
#  'vertices_multiplicity'
]

simcandkeys = [
 'simTICLCandidate_raw_energy',
 'simTICLCandidate_regressed_energy',
 # 'simTICLCandidate_simTracksterCPIndex',
 # 'simTICLCandidate_boundaryX',
 # 'simTICLCandidate_boundaryY',
 # 'simTICLCandidate_boundaryZ',
 # 'simTICLCandidate_boundaryPx',
 # 'simTICLCandidate_boundaryPy',
 # 'simTICLCandidate_boundaryPz',
 # 'simTICLCandidate_time',
 # 'simTICLCandidate_caloParticleMass',
 # 'simTICLCandidate_pdgId',
 # 'simTICLCandidate_charge',
 'simTICLCandidate_track_in_candidate']

candkeys = [
 # 'NCandidates',
 'candidate_charge',
 'candidate_pdgId',
 'candidate_id_probabilities',
 'candidate_time',
 'candidate_timeErr',
 'candidate_energy',
 'candidate_raw_energy',
 # 'candidate_px',
 # 'candidate_py',
 # 'candidate_pz',
 'track_in_candidate',
 'tracksters_in_candidate']

PT = int(sys.argv[1])
ETA = float(sys.argv[2])
label = "pt"+str(PT)+"_eta"+str(ETA).replace(".","p")
file = uproot.open("/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_"+label+"/histo_"+label+".root")

print("opening file", label)

alltracksters = load_branch_with_highest_cycle(file,'ticlDumper/ticlTrackstersCLUE3DHigh')
allsimtrackstersCP = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

simtrackstersCP = allsimtrackstersCP.arrays(simTsKeys)
simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
# tracksters = alltracksters.arrays(tsKeys)
associations = allassociations.arrays(assKeys)
tracks = alltracks.arrays(tracksKeys)
tracksterLinks = allticlTracksterLinks.arrays(tsKeys)

#track_pt = tracks.track_pt
#track_id = tracks.track_id
#track_hgcal_eta = tracks.track_hgcal_eta
#track_hgcal_phi = tracks.track_hgcal_phi
#track_hgcal_pt = tracks.track_hgcal_pt
#track_missing_outer_hits = tracks.track_missing_outer_hits
#track_missing_inner_hits = tracks.track_missing_inner_hits
#track_nhits = tracks.track_nhits
#track_quality = tracks.track_quality
#track_time_mtd_err = tracks.track_time_mtd_err
#track_isMuon = tracks.track_isMuon
#track_isTrackerMuon = tracks.track_isTrackerMuon

## track_boundaryX = simtrackstersSC["track_boundaryX"].array()
## track_boundaryY = simtrackstersSC["track_boundaryY"].array()
## track_boundaryZ = simtrackstersSC["track_boundaryZ"].array()

#tracks_p = tracks.track_p
tracks_time = tracks.track_time_mtd
tracks_timeErr = tracks.track_time_mtd_err
tracks_MTDposX = tracks['track_pos_mtd/track_pos_mtd.theVector.theX']
tracks_MTDposY = tracks['track_pos_mtd/track_pos_mtd.theVector.theY']
tracks_MTDposZ = tracks['track_pos_mtd/track_pos_mtd.theVector.theZ']

rEp_ass_all = []
rEp_other_all = []

pullEp_ass_all = []
pullEp_other_all = []

dR_ass_all = []
dR_other_all = []

pull_dir_ass_all = []
pull_dir_other_all = []

delta_t_ass_all = []
delta_t_other_all = []

pull_time_ass_all = []
pull_time_other_all = []

print("Compute residuals and pulls")

for ev in tqdm(prange(len(simtrackstersSC))):
    stsSCEv = simtrackstersSC[ev]
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    assEv = associations[ev]
#     print("--- ", ev, " ---")
    for idx in prange(len(stsSCEv.trackIdx)):
        # find track position in the tracks array using track index
        trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
        if trk_id == -1:
            continue

        refEta = tracksEv.track_hgcal_eta[trk_id]
        refPhi = tracksEv.track_hgcal_phi[trk_id]
        refPt = tracksEv.track_hgcal_pt[trk_id]
        refP = tracksEv.track_p[trk_id]

        ########## selecting good tracksters
        maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.9
        tsEnergy = tsLinksEv.raw_energy[assEv.ticlTracksterLinks_simToReco_SC[idx]]
        sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
        maskEnergy = sharedEnergy / tsEnergy > 0.4
        maskIdx = maskScore & maskEnergy
        assIndices = assEv.ticlTracksterLinks_simToReco_SC[idx][maskIdx]
        sharedEnergy = sharedEnergy[maskScore & maskEnergy]
        ######### mask on the other tracksters
        n_ts = len(tsLinksEv.barycenter_eta)
        assoc_mask = np.ones(n_ts, dtype=bool)
        assoc_mask[assIndices] = False
        sameSide = tsLinksEv.barycenter_eta * refEta > 0
        other_mask = sameSide & assoc_mask
        #########

        if not(len(assIndices)): continue # no associated tracksters

        allTsEta = tsLinksEv.barycenter_eta[other_mask]
        allTsPhi = tsLinksEv.barycenter_phi[other_mask]
        allTsZ = tsLinksEv.barycenter_z[other_mask]
        allTsEnergy = tsLinksEv.raw_energy[other_mask]

        shareTsEta = tsLinksEv.barycenter_eta[assIndices]
        shareTsPhi = tsLinksEv.barycenter_phi[assIndices]
        shareTsZ = tsLinksEv.barycenter_z[assIndices]
        tsEnergy = tsLinksEv.raw_energy[assIndices]

        isEM_other = np.abs(allTsZ) < 368
        f_other = np.where(isEM_other, 1.0, 0.7)
        rEp_other = allTsEnergy - f_other * refP;
        pullEp_other = rEp_other / sigmaEp(refP, isEM_other)

        isEM_ass = np.abs(shareTsZ) < 368
        f_ass = np.where(isEM_ass, 1.0, 0.7)
        rEp_ass = tsEnergy - f_ass * refP
        pullEp_ass = rEp_ass / sigmaEp(refP, isEM_ass)

#         print("-- ts --")

        dR_ass = distWrap_numba(refEta, refPhi, shareTsEta, shareTsPhi)
        dR_other = distWrap_numba(refEta, refPhi, allTsEta, allTsPhi)

        sigma_dir = 0.02  # roughly 20 mrad
        pull_dir_ass = dR_ass / sigma_dir
        pull_dir_other = dR_other / sigma_dir
        #0.02 ≈ typical HGCAL shower axis resolution in radians
        #You can adjust later using MC distributions (same as we did for 3D angles)

        trk_timeErr = tracks_timeErr[ev][trk_id]
        trk_time = tracks_time[ev][trk_id]
        trk_MTDposX = tracks_MTDposX[ev][trk_id]
        trk_MTDposY = tracks_MTDposY[ev][trk_id]
        trk_MTDposZ = tracks_MTDposZ[ev][trk_id]

        tsTime = tsLinksEv.time
        tsTimeError = tsLinksEv.timeError
        ts_x = tsLinksEv.barycenter_x
        ts_y = tsLinksEv.barycenter_y
        ts_z = tsLinksEv.barycenter_z

        valid_time = (trk_timeErr > 0) & (tsTimeError > 0)

        tof = distance(trk_MTDposX, trk_MTDposY, trk_MTDposZ, ts_x, ts_y, ts_z) / C_CM_PER_NS
        delta_t = ak.where(valid_time, tsTime - trk_time - tof, 0.0)

        delta_t_ass   = delta_t[assIndices]
        delta_t_other = delta_t[other_mask]

        sigma_time = np.sqrt(trk_timeErr**2 + tsTimeError**2)

        pull_ass = delta_t_ass / sigma_time[assIndices]
        pull_other = delta_t_other / sigma_time[other_mask]

        # E/p
        rEp_ass_all.extend(ak.to_numpy(rEp_ass))
        rEp_other_all.extend(ak.to_numpy(rEp_other))

        pullEp_ass_all.extend(ak.to_numpy(pullEp_ass))
        pullEp_other_all.extend(ak.to_numpy(pullEp_other))

        # Direction
        dR_ass_all.extend(ak.to_numpy(dR_ass))
        dR_other_all.extend(ak.to_numpy(dR_other))

        pull_dir_ass_all.extend(ak.to_numpy(pull_dir_ass))
        pull_dir_other_all.extend(ak.to_numpy(pull_dir_other))

        # Time
        delta_t_ass_all.extend(ak.to_numpy(delta_t_ass))
        delta_t_other_all.extend(ak.to_numpy(delta_t_other))

        pull_time_ass_all.extend(ak.to_numpy(pull_ass))
        pull_time_other_all.extend(ak.to_numpy(pull_other))

def arr(x):
    return np.asarray(x, dtype=float)

rEp_ass = arr(rEp_ass_all)
rEp_oth = arr(rEp_other_all)
pEp_ass = arr(pullEp_ass_all)
pEp_oth = arr(pullEp_other_all)
dR_ass = arr(dR_ass_all)
dR_oth = arr(dR_other_all)
pR_ass = arr(pull_dir_ass_all)
pR_oth = arr(pull_dir_other_all)
dt_ass = arr(delta_t_ass_all)
dt_oth = arr(delta_t_other_all)
pt_ass = arr(pull_time_ass_all)
pt_oth = arr(pull_time_other_all)

sigma_rEp = np.std(rEp_ass)
sigma_pullR = np.std(pR_ass)

rEp_ass_norm = rEp_ass / sigma_rEp
pR_ass_norm  = pR_ass / sigma_pullR

rEp_oth_norm = rEp_oth / sigma_rEp
pR_oth_norm  = pR_oth / sigma_pullR

SIGMA_REP = sigma_rEp

plt.style.use(hep.style.CMS)

# -------------------------
# 1) E/p residuals
# -------------------------
plt.figure(figsize=(30, 10))
plt.suptitle("E/p")
plt.subplot(1, 3, 1)
plt.hist(rEp_ass, bins=100, range=(-150,50), density=True, label="associated")
plt.hist(rEp_oth, bins=100, range=(-150,50), density=True, histtype="step", label="other")
plt.xlabel(r"$E - f(p)\,p$  [GeV]")
plt.ylabel("Density")
plt.title("residuals")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.hist(pEp_ass, bins=100, range=(-6,6), density=True, label="associated")
plt.hist(pEp_oth, bins=100, range=(-6,6), density=True, histtype="step", label="other")
plt.xlabel(r"$(E - f(p)\,p)/\sigma_{E/p}$")
plt.ylabel("Density")
plt.title("pulls")
plt.legend()
plt.grid()

plt.grid()
plt.subplot(1, 3, 3)
plt.hist(rEp_ass_norm, bins=100, range=(-3,1), density=True, label="associated")
plt.hist(rEp_oth_norm, bins=100, range=(-3,1), density=True, histtype="step", label="other")
plt.xlabel(r"$E - f(p)\,p/\sigma_{E - f(p)\,p}$")
plt.ylabel("Density")
plt.title(f"residuals normalized ($\sigma$={sigma_rEp:.2f})")
plt.legend()
plt.savefig("plotsNew/residualsEp"+label+".png")

# -------------------------
# 2) Angular consistency
# -------------------------
plt.figure(figsize=(20, 10))
plt.suptitle("dR")
plt.subplot(1, 2, 1)
plt.hist(dR_ass, bins=100, range=(0,0.2), density=True, label="associated")
plt.hist(dR_oth, bins=100, range=(0,0.2), density=True, histtype="step", label="other")
plt.xlabel(r"$\Delta R(\mathrm{track}, \mathrm{trackster})$")
plt.ylabel("Density")
plt.legend()
plt.title("residuals")
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(pR_ass, bins=100, range=(0,5), density=True, label="associated")
plt.hist(pR_oth, bins=100, range=(0,5), density=True, histtype="step", label="other")
plt.xlabel(r"$\Delta R / \sigma_{\mathrm{dir}}$")
plt.ylabel("Density")
plt.legend()
plt.title("pulls")
plt.grid()
plt.savefig("plotsNew/residualsR"+label+".png")

# # -------------------------
# # 3) Timing consistency
# # -------------------------
plt.figure(figsize=(20, 10))
plt.suptitle("dTime")
plt.subplot(1, 2, 1)
plt.hist(dt_ass, bins=100, range=(-2,2), density=True, label="associated")
plt.hist(dt_oth, bins=100, range=(-2,2), density=True, histtype="step", label="other")
plt.xlabel(r"$\Delta t = t_{\mathrm{TS}} - t_{\mathrm{trk}} - \mathrm{TOF}$  [ns]")
plt.ylabel("Density")
plt.legend()
plt.title("residuals")
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(pt_ass, bins=100, range=(-5,5), density=True, label="associated")
plt.hist(pt_oth, bins=100, range=(-5,5), density=True, histtype="step", label="other")
plt.xlabel(r"$\Delta t / \sigma_t$")
plt.ylabel("Density")
plt.legend()
plt.title("pulls")
plt.grid()
plt.savefig("plotsNew/residualsTime"+label+".png")

# cost function

print("ROC time")

time_valid_ass = pt_ass != 0
time_valid_oth = pt_oth != 0
n_terms_ass = 2 + time_valid_ass
n_terms_oth = 2 + time_valid_oth

# ## ROC and sanity checks

from sklearn.metrics import roc_curve, auc

def build_score(wEp, rEp_ass_norm, rEp_oth_norm, pR_ass, pR_oth):
    wR = 1 - wEp
    score_ass = ( wEp * (rEp_ass_norm)**2 + wR * (pR_ass)**2 )
    score_oth = ( wEp * (rEp_oth_norm)**2 + wR * (pR_oth)**2 )

    y_true = np.concatenate([
        np.ones_like(score_ass),     # matched
        np.zeros_like(score_oth)     # unmatched
    ])

    y_score = np.concatenate([
        score_ass,
        score_oth
    ])

    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def build_score_time(wEp, wT, rEp_ass_norm, rEp_oth_norm, pR_ass, pR_oth, pt_ass, pt_oth):
    wR = 1 - wEp

    time_valid_ass = pt_ass != 0
    time_valid_oth = pt_oth != 0
    n_terms_ass = 2 + time_valid_ass
    n_terms_oth = 2 + time_valid_oth
    score_ass = ( wEp * (rEp_ass_norm)**2 + wR * (pR_ass)**2 + wT * (pt_ass)**2) / n_terms_ass
    score_oth = ( wEp * (rEp_oth_norm)**2 + wR * (pR_oth)**2 + wT * (pt_oth)**2) / n_terms_oth

    y_true = np.concatenate([
        np.ones_like(score_ass),     # matched
        np.zeros_like(score_oth)     # unmatched
    ])

    y_score = np.concatenate([
        score_ass,
        score_oth
    ])

    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def plot_ROC(fpr, tpr, roc_auc, wEp):
    plt.plot(fpr, tpr, label=f"AUC wEp="+ str(wEp) + f" = {roc_auc:.6f}")
    plt.plot([0,1], [0,1], "k--", lw=1)

def plot_ROC_time(fpr, tpr, roc_auc, wEp, wT):
    plt.plot(fpr, tpr, label=f"AUC wEp="+ str(wEp) + " wT=" + str(wT) + f" = {roc_auc:.6f}")
    plt.plot([0,1], [0,1], "k--", lw=1)

plt.figure(figsize=(20, 10))
plt.suptitle("Total cost")
plt.subplot(1, 2, 1)
for wEp in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    fpr, tpr, roc_auc = build_score(wEp, rEp_ass_norm, rEp_oth_norm, pR_ass, pR_oth)
    plot_ROC(fpr, tpr, roc_auc, wEp)
plt.xlabel("False Positive Rate (unmatched)")
plt.ylabel("True Positive Rate (matched)")
plt.title("AUC score without time info")
plt.legend()
plt.grid()
plt.grid()

plt.subplot(1, 2, 2)
for wEp in [0.4, 0.8]:
    for wT in [0, 0.5, 1]:
        fpr, tpr, roc_auc = build_score_time(wEp, wT, rEp_ass_norm, rEp_oth_norm, pR_ass, pR_oth, pt_ass, pt_oth)
        plot_ROC_time(fpr, tpr, roc_auc, wEp, wT)
plt.xlabel("False Positive Rate (unmatched)")
plt.ylabel("True Positive Rate (matched)")
plt.title("AUC score with time info")
plt.legend()
plt.grid()
plt.savefig("plotsNew/roc_auc_"+label+".png")

wEp = 0.2
wR = 1 - wEp
wT = 0.5

score_ass = ( wEp * (rEp_ass_norm)**2 + wR * (pR_ass)**2 + wT * (pt_ass)**2) / n_terms_ass
score_oth = ( wEp * (rEp_oth_norm)**2 + wR * (pR_oth)**2 + wT * (pt_oth)**2) / n_terms_oth

score_noTime_ass = ( wEp * (rEp_ass_norm)**2 + wR * (pR_ass)**2 )
score_noTime_oth = ( wEp * (rEp_oth_norm)**2 + wR * (pR_oth)**2 )

plt.style.use(hep.style.CMS)

plt.figure(figsize=(20, 10))
plt.suptitle("Total cost")
plt.subplot(1, 2, 1)
CUT = 5
n_ass = plt.hist(score_ass, bins=100, range=(0,CUT), color = "dodgerblue", label=f"associated (over= {len(score_ass[score_ass>CUT])})")
n_oth = plt.hist(score_oth, bins=100, range=(0,CUT), color = "red", histtype="step", label=f"other (over= {len(score_oth[score_oth>CUT])})")
THR=1
plt.text(CUT/4, max(max(n_ass[0]), max(n_oth[0]))*2/3, f"CUT at {THR}:\n{len(score_ass[score_ass<THR])/len(score_ass)*100:.0f}% of ass ({len(score_ass[score_ass<THR])})\n{len(score_oth[score_oth<THR])/len(score_oth)*100:.3f}% of others ({len(score_oth[score_oth<THR])})")
plt.xlabel("score")
plt.ylabel("Counts")
plt.title("Cost")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
CUT = 5
n_ass = plt.hist(score_noTime_ass, bins=100, range=(0,CUT), color = "dodgerblue", label=f"associated (over= {len(score_noTime_ass[score_noTime_ass>CUT])})")
n_oth = plt.hist(score_noTime_oth, bins=100, range=(0,CUT), color = "red", histtype="step", label=f"other (over= {len(score_noTime_oth[score_noTime_oth>CUT])})")
THR=2
plt.text(CUT/4, max(max(n_ass[0]), max(n_oth[0]))*2/3, f"CUT at {THR}:\n{len(score_noTime_ass[score_noTime_ass<THR])/len(score_noTime_ass)*100:.0f}% of ass ({len(score_noTime_ass[score_noTime_ass<THR])})\n"+
               f"{len(score_noTime_oth[score_noTime_oth<THR])/len(score_noTime_oth)*100:.3f}% of others ({len(score_noTime_oth[score_noTime_oth<THR])})")
plt.xlabel("Score")
plt.ylabel("Counts")
plt.title("Cost without time")
plt.legend()
plt.grid()
plt.savefig("plotsNew/finalCost_"+label+".png")

Ep = []
pt = []
eta = []

for ev in tqdm(prange(len(simtrackstersSC))):
    stsSCEv = simtrackstersSC[ev]
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    assEv = associations[ev]
#     print("--- ", ev, " ---")
    for idx in prange(len(stsSCEv.trackIdx)):
        # find track position in the tracks array using track index
        trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
        if trk_id == -1:
            continue

        refEta = tracksEv.track_hgcal_eta[trk_id]
        refPhi = tracksEv.track_hgcal_phi[trk_id]
        refPt = tracksEv.track_hgcal_pt[trk_id]
        refP = tracksEv.track_p[trk_id]

        ########## selecting good tracksters
        stsRawEnergy = stsSCEv.raw_energy[idx]
        stsEnergy = stsSCEv.regressed_energy[idx]
        if abs(stsSCEv.pdgID[idx]) == 211:
            pt.append(refPt)
            eta.append(refEta)
            Ep.append(stsRawEnergy/stsEnergy)


plt.figure(figsize=(20, 10))
plt.suptitle("simTs rawEnergy/regressedEnergy, mean:" + str(np.mean(Ep)))
plt.subplot(1, 2, 1)
plt.hist2d(pt, Ep, range=[[PT-5, PT+5],[0,1.5]], bins=50, cmap="hot_r")
plt.xlabel(r"$p_T [Gev]$")
plt.ylabel("E/p")
plt.title("E/p for HAD objects")
plt.grid()

plt.subplot(1, 2, 2)
plt.hist2d(np.abs(eta), Ep, bins=50, range=[[ETA-0.2, ETA+0.2], [0, 1.5]], cmap="hot_r")
plt.xlabel(r"$\eta$")
plt.ylabel("E/p")
plt.title("E/p for HAD objects")
plt.grid()
plt.savefig("plotsNew/f_p_"+label+".png")

#####################
# tiles and linking #
#####################

print("Compute tiles and associations")

class EtaPhiTiles:
    def __init__(self, eta_min, eta_max, n_eta=34, phi_min=-math.pi, phi_max=math.pi, n_phi=126):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        self.n_eta = n_eta
        self.n_phi = n_phi

        self.d_eta = (eta_max - eta_min) / n_eta
        self.d_phi = (phi_max - phi_min) / n_phi

        # 2D grid of lists
        self.tiles = [[[] for _ in range(n_phi)] for _ in range(n_eta)]

    def _eta_bin(self, eta):
        i = int((eta - self.eta_min) / self.d_eta)
        return max(0, min(self.n_eta - 1, i))

    def _phi_bin(self, phi):
        # wrap phi into [-pi, pi)
        while phi < -math.pi:
            phi += 2 * math.pi
        while phi >= math.pi:
            phi -= 2 * math.pi

        i = int((phi - self.phi_min) / self.d_phi)
        return max(0, min(self.n_phi - 1, i))

    def fill(self, eta, phi, idx):
        i_eta = self._eta_bin(eta)
        i_phi = self._phi_bin(phi)
        self.tiles[i_eta][i_phi].append(idx)

    def get_tile(self, eta, phi):
        i_eta = self._eta_bin(eta)
        i_phi = self._phi_bin(phi)
        return self.tiles[i_eta][i_phi]

    def get_window(self, eta, phi, d_eta_bins=1, d_phi_bins=1):
        """Return all indices in neighboring bins"""
        i_eta = self._eta_bin(eta)
        i_phi = self._phi_bin(phi)

        out = []

        for ie in range(i_eta - d_eta_bins, i_eta + d_eta_bins + 1):
            if ie < 0 or ie >= self.n_eta:
                continue

            for ip in range(i_phi - d_phi_bins, i_phi + d_phi_bins + 1):
                ip_wrapped = ip % self.n_phi  # φ wraps
                out.extend(self.tiles[ie][ip_wrapped])

        return out

def propagate_tracksters(barycenters, zVal):
    """
    barycenters: (N,3) array
    returns: eta, phi, endcap
    """
    baryc = barycenters
    norms = np.linalg.norm(baryc, axis=1)
    direct = baryc / norms[:, None]

    z = np.where(baryc[:, 2] > 0, zVal, -zVal)
    par = (z - baryc[:, 2]) / direct[:, 2]

    x = par * direct[:, 0] + baryc[:, 0]
    y = par * direct[:, 1] + baryc[:, 1]

    r = np.sqrt(x*x + y*y + z*z)
    eta = 0.5 * np.log((r + z) / (r - z))
    phi = np.arctan2(y, x)

    endcap = np.where(eta > 0, 1, 0)

    return eta, phi, endcap

# no diff in eta and phi if we use barycenter for propagation instead of PCA
# ev = 0
# tracksEv = tracks[ev]
# tsLinksEv = tracksterLinks[ev]
# #for idx in prange(len(tsLinksEv.raw_energy)):

# refEta = tsLinksEv.barycenter_eta
# refPhi = tsLinksEv.barycenter_phi
# refX = tsLinksEv.barycenter_x
# refY = tsLinksEv.barycenter_y
# refZ = tsLinksEv.barycenter_z
# barycenters= np.asarray([refX, refY, refZ]).T

# zVal = 322
# eta, phi, endcap = propagate_tracksters(barycenters, zVal)
# if ak.any(np.abs(eta-refEta)> 1e-5): print("diff eta")
# if ak.any(np.abs(phi-refPhi)> 1e-5): print("diff phi")

def fill_tiles(eta, phi, endcap, tracksterTiles):
    for idx in range(len(eta)):
        ec = endcap[idx]
        tracksterTiles[ec].fill(eta[idx], phi[idx], idx)

def find_candidates_for_tracks(trk_id, eta_trk, phi_trk, endcap_trk, tracksterTiles, d_eta_bins=1, d_phi_bins=1):
    all_candidates = {}

    for i in range(len(eta_trk)):
        ec = endcap_trk[i]
        cands = tracksterTiles[ec].get_window(
            eta_trk[i],
            phi_trk[i],
            d_eta_bins,
            d_phi_bins
        )
        all_candidates[trk_id[i]] = cands

    return all_candidates

all_candidates = []
for ev in tqdm(prange(len(tracks))):
    n_eta = 30
    tracksterTiles = [
        EtaPhiTiles(eta_min=-3.0, eta_max=-1.5, n_eta=n_eta),
        EtaPhiTiles(eta_min= 1.5, eta_max= 3.0, n_eta=n_eta),
    ]

    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    #for idx in prange(len(tsLinksEv.raw_energy)):

    refPt = tracksEv.track_hgcal_pt
    refP = tracksEv.track_p
    mask = np.logical_and(refPt > 1, refP > 2)

    refPt = refPt[mask]
    refP = refP[mask]
    refEta = tracksEv.track_hgcal_eta[mask]
    refPhi = tracksEv.track_hgcal_phi[mask]
    refId = tracksEv.track_id[mask]

    refEndcap = np.where(refEta>0, 1, 0)

    tsEta = tsLinksEv.barycenter_eta
    tsPhi = tsLinksEv.barycenter_phi
    tsEnergy = tsLinksEv.raw_energy
    tsEndcap = np.where(tsEta>0, 1, 0)
#     tsX = tsLinksEv.barycenter_x
#     tsY = tsLinksEv.barycenter_y
    tsZ = tsLinksEv.barycenter_z
#     barycenters= np.asarray([tsX, tsY, tsZ]).T
#     zVal = 322
#     eta, phi, endcap = propagate_tracksters(barycenters, zVal)
    fill_tiles(tsEta, tsPhi, tsEndcap, tracksterTiles)
    all_candidates_ev = find_candidates_for_tracks(refId, refEta, refPhi, refEndcap, tracksterTiles)
    all_candidates.append(all_candidates_ev)

def greedy_global_linking(all_scores, score_cut=2.0, loose_cut=10):
    # sort by best score first and pt then
    #all_scores.sort(key=lambda x: (x[0], -tracksEv.track_hgcal_pt[x[1]]))

    used_tracksters = set()
    links = {}
    links_no_score = {}
#     print(all_scores)
    all_scores_filt = [t for t in all_scores if t[0] < loose_cut]
    for score, itrk, its in all_scores_filt:
        #if score > loose_cut:
        #    break

#         print("ts", its, its in used_tracksters)
        if its in used_tracksters:
            continue

        used_tracksters.add(its)

        new = False
        if itrk not in links:
            links[itrk] = []
            links_no_score[itrk] = []
            new = True
        if score < score_cut or new:
            links[itrk].append((its, score))
            links_no_score[itrk].append(its)

    return links, links_no_score

@njit
def func(x, y):
    y = np.abs(y)
    a = 2.29516386e-01
    b = -8.05371532e+02
    c = -6.45573586e-01
    d = 8.05370082e+02
    e = 1.00000033e+00
    f = -1.07458042e-04
    g = 4.79631755e-01
    h = 1.21330763e+00
    if x>200: x=200
    if y<1.7: y= 1.7
    val = a + b*x + c*y + d*x**e + f*x*y + g * y**h
    if val < 0.008: val =0.008
    return val

@njit
def compute_score_block(refPt,
    refEta, refPhi, refP,
    tsEta, tsPhi, tsEnergy, tsZ,
    score_out, wEp=0.5 #2
):
    n = len(tsEta)

    for i in range(n):
        # isEM + f
        f = 1.0 if abs(tsZ[i]) < 368.0 else 0.7

        rEp = (tsEnergy[i] - refP) / refP # - f *
        rEp_norm = rEp #/ SIGMA_REP # FIXME

        # dR with phi wrapping
        deltaPhi = refPhi - tsPhi[i]
        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
        deta = refEta - tsEta[i]
        dR2 = deta ** 2 + deltaPhi ** 2

        pullR2 = dR2 / func(refPt, refEta)**2 # 0.02**2

        score_out[i] = wEp * rEp_norm*rEp_norm + (1-wEp) * pullR2 / 2


print("Compute Links")

all_links = []
all_links_no_score = []
all_scores = []

CUT = 10
for ev in tqdm(prange(len(tracks))):
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]

    has_cand = np.array([
        len(all_candidates[ev].get(i, [])) > 0
        for i in tracksEv.track_id
    ])

    valid_tracks = np.where(
        (tracksEv.track_hgcal_pt >= 1.0) &
        (tracksEv.track_p >= 2.0)
        &  has_cand
    )[0]

    refEtaAll = tracksEv.track_hgcal_eta
    refPhiAll = tracksEv.track_hgcal_phi
    refPtAll = tracksEv.track_hgcal_pt
    refPAll = tracksEv.track_p
    refIdAll = tracksEv.track_id
    tsEtaAll    = tsLinksEv.barycenter_eta
    tsPhiAll    = tsLinksEv.barycenter_phi
    tsEnergyAll = tsLinksEv.raw_energy
    tsZAll      = tsLinksEv.barycenter_z
    all_candidates_ev = all_candidates[ev]

    refPtValid = tracksEv.track_hgcal_pt[valid_tracks]
    idx_sorted = np.argsort(-refPtValid)
    sorted_valid_tracks = valid_tracks[idx_sorted]

    all_scores_ev = []
    all_scores_no_cut = []
    for idx in sorted_valid_tracks:

        refEta = refEtaAll[idx]
        refPhi = refPhiAll[idx]
        refPt  = tracksEv.track_hgcal_pt[idx]
        refP   = refPAll[idx]
        refId  = refIdAll[idx]

        all_candidates_trk = all_candidates_ev[refId]

        tsEta = tsEtaAll[all_candidates_trk]
        tsPhi = tsPhiAll[all_candidates_trk]
        tsEnergy = tsEnergyAll[all_candidates_trk]
        tsZ = tsZAll[all_candidates_trk]

        score = np.empty(len(tsEta), dtype=np.float32)

        compute_score_block(refPt,
            refEta, refPhi, refP,
            tsEta, tsPhi, tsEnergy, tsZ,
            score, wEp = 0.5
        )
        n = len(all_candidates_trk)

        all_scores_ev.extend(
            zip(
                score[score<CUT].tolist(),
                [refId] * n,
                np.array(all_candidates_trk)[score<CUT]
            )
        )
        all_scores_no_cut.extend(
            zip(
                score.tolist(),
                [refId] * n,
                all_candidates_trk
            )
        )

    links, links_no_score = greedy_global_linking(all_scores_ev, score_cut=1, loose_cut=2)
#     print(links)
    all_scores.append(all_scores_no_cut)
    all_links.append(links)
    all_links_no_score.append(links_no_score)

print("True Links")
# true links
true_links = []
for ev in tqdm(prange(len(simtrackstersSC))):
    stsSCEv = simtrackstersSC[ev]
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    assEv = associations[ev]
    true_links_ev = []
    for idx in prange(len(stsSCEv.trackIdx)):
        # find track position in the tracks array using track index
        trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
        if trk_id == -1:
            continue
        refEta = tracksEv.track_hgcal_eta[trk_id]
        refPhi = tracksEv.track_hgcal_phi[trk_id]
        refPt = tracksEv.track_hgcal_pt[trk_id]
        refP = tracksEv.track_p[trk_id]

        ########## selecting good tracksters
        maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.9
        tsEnergy = tsLinksEv.raw_energy[assEv.ticlTracksterLinks_simToReco_SC[idx]]
        sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
        maskEnergy = sharedEnergy / tsEnergy > 0.4
        maskIdx = maskScore & maskEnergy
        assIndices = assEv.ticlTracksterLinks_simToReco_SC[idx][maskIdx]

        true_links_ev.append((stsSCEv.trackIdx[idx], assIndices))

    true_links.append(true_links_ev)

effTight_flags = []
effLoose_flags = []
purity_flags = []

wrong = []
for ev in prange(len(true_links)):

    links_ev = all_links_no_score[ev]
    tracksEv = tracks[ev]
    for itrk, true_ts_list in true_links[ev]:
        if not len(true_ts_list): continue
        trk_pos = find_track_id(tracksEv.track_id, itrk)
        if trk_pos == -1:
            continue

        # Track kinematics
        eta = abs(tracksEv.track_hgcal_eta[trk_pos])
        pt  = tracksEv.track_hgcal_pt[trk_pos]
        p  = tracksEv.track_p[trk_pos]

        if (eta < ETA-0.05 or eta > ETA+0.05) or (pt < PT*0.75 or pt > PT*1.25): continue

        if pt < 1 or p < 2: continue

        # Was this track linked at all?
        try:
            reco_ts = links_ev[itrk]
        except:
            effTight_flags.append(False)
            effLoose_flags.append(False)
            purity_flags.append(False)
            wrong.append((ev, itrk))
            continue

        # True if ANY true trackster is found
        matched_loose = any(its in reco_ts for its in true_ts_list)
        matched_tight = all(its in reco_ts for its in true_ts_list)
        pure = all(its in true_ts_list for its in reco_ts)

        effTight_flags.append(matched_tight)
        effLoose_flags.append(matched_loose)
        purity_flags.append(pure)

# print(effTight_flags)
effTight = np.mean(arr(effTight_flags))
effLoose = np.mean(arr(effLoose_flags))
purity = np.mean(arr(purity_flags))

# Append mode: adds to the end of the file
file1 = open("plotsNew/efficiencies.txt", "a")
file1.write(str(PT) + " " + str(ETA) + " " + str(effTight) + " " + str(effLoose) + " " + str(purity)+'\n')
file1.close()

# debug

print("debug")

def debug_missing_links(ev, trk_id, File):
    # find the tuple for this track
    correct_ts_ids = None
    for tid, ts_arr in true_links[ev]:
        if tid == trk_id:
            correct_ts_ids = np.asarray(ts_arr)
            break

    if correct_ts_ids is None:
        File.write(f"TYPE 1: Track {trk_id} not found in true_links for event {ev}\n")
        return

    # structured array of scores
    scores_to_arr = np.array(
        all_scores[ev],
        dtype=[('score', float), ('trk', np.uint32), ('ts', np.int64)]
    )

    # all candidates for this track
    matches = scores_to_arr[scores_to_arr['trk'] == trk_id]

    File.write(f"\nEvent {ev}, track {trk_id}\n")
    File.write(f"Correct TS IDs: {correct_ts_ids.tolist()}\n")

    for correct_ts in correct_ts_ids:
        candidate_entry = matches[matches['ts'] == correct_ts]
        if len(candidate_entry) == 0:
            File.write(f"  TYPE 4: Correct TS {correct_ts} not in candidate window\n")
            continue

        score_correct = candidate_entry['score'][0]
        File.write(f"  Correct TS {correct_ts} candidate score: {score_correct:.3f}\n")

        # check if linked elsewhere
        ts_linked = None
        linked_score = None
        for trk, linked_list in all_links[ev].items():
            for ts_id, sc in linked_list:
                if ts_id == correct_ts:
                    ts_linked = trk
                    linked_score = sc
                    break
            if ts_linked is not None:
                break

        if ts_linked is None:
            File.write(f"    TYPE 2: Correct TS {correct_ts} not linked\n")
        else:
            File.write(
                f"    TYPE 3: Correct TS {correct_ts} wrongly linked to track "
                f"{ts_linked} (score: {linked_score:.3f})\n"
            )

            # get pT/P for correct and wrongly linked tracks
            pos_correct = np.where(tracks[ev].track_id == trk_id)[0][0]
            Pt_correct = tracks[ev].track_hgcal_pt[pos_correct]
            P_correct  = tracks[ev].track_p[pos_correct]

            pos_wrong = np.where(tracks[ev].track_id == ts_linked)[0][0]
            Pt_wrong = tracks[ev].track_hgcal_pt[pos_wrong]
            P_wrong  = tracks[ev].track_p[pos_wrong]

            File.write(
                f"      Correct track Pt: {Pt_correct:.3f}, P: {P_correct:.3f} | "
                f"Wrongly linked track Pt: {Pt_wrong:.3f}, P: {P_wrong:.3f}\n"
            )

with open("plotsNew/missing_links_"+label+".txt", "w") as f:
    for ev, trk in wrong:
        debug_missing_links(ev=ev, trk_id=trk, File=f)
