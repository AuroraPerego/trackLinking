import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib

from numba import prange, njit
import awkward.numba

plt.style.use(hep.style.CMS)

def arr(x):
    return np.asarray(x, dtype=float)

@njit
def flatten_numba(a):
    return [x[0] if len(x) else 0 for x in a]

def dist(refEta, refPhi, otherTsEta, otherTsPhi):
    return ((otherTsEta-refEta)**2 + (otherTsPhi-refPhi)**2)**0.5

def distWrap2(refEta, refPhi, otherTsEta, otherTsPhi):
    deltaPhi = otherTsPhi - refPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    return ((otherTsEta - refEta) ** 2 + deltaPhi ** 2)

@njit
def distWrap2_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    deltaPhi = otherTsPhi - refPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    return ((otherTsEta - refEta) ** 2 + deltaPhi ** 2)

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

C = 29.9792458 #cm/ns

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
 'CPidx',
 'pdgID',
 'vertices_indexes',
 'vertices_x',
 'vertices_y',
 'vertices_z',
 'vertices_time',
#  'vertices_timeErr',
 'vertices_energy',
 'vertices_multiplicity'
]

assKeys = [
 'ticlTracksterLinks_recoToSim_SC',
 'ticlTracksterLinks_recoToSim_SC_score',
 'ticlTracksterLinks_recoToSim_SC_sharedE',
 'ticlTracksterLinks_simToReco_SC',
 'ticlTracksterLinks_simToReco_SC_score',
 'ticlTracksterLinks_simToReco_SC_sharedE',
#     'ticlCandidate_simToReco_CP_score',
#     'ticlCandidate_simToReco_CP_sharedE'
 'ticlCandidate_simToReco_SC',
 'ticlCandidate_simToReco_SC_score',
 'ticlCandidate_simToReco_SC_sharedE',
 'ticlCandidate_recoToSim_SC',
 'ticlCandidate_recoToSim_SC_score',
 'ticlCandidate_recoToSim_SC_sharedE',
#  'ticlCandidate_simToReco_CP',
#  'ticlCandidate_simToReco_CP_score',
#  'ticlCandidate_simToReco_CP_sharedE',
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
 'vertices_x',
 'vertices_y',
 'vertices_z',
#  'vertices_time',
#  'vertices_timeErr',
 'vertices_energy',
#  'vertices_correctedEnergy',
#  'vertices_correctedEnergyUncertainty',
#  'vertices_multiplicity'
]

class EtaPhiTiles:
    def __init__(self, eta_min, eta_max, n_eta=34, phi_min=-np.pi, phi_max=np.pi, n_phi=126):
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

    def getEtaWidth(self):
        return self.d_eta

    def getPhiWidth(self):
        return self.d_phi

    def getNumEta(self):
        return self.n_eta

    def getNumPhi(self):
        return self.n_phi

    def _eta_bin(self, eta):
        i = int((eta - self.eta_min) / self.d_eta)
        return max(0, min(self.n_eta - 1, i))

    def _phi_bin(self, phi):
        # wrap phi into [-pi, pi)
        while phi < -np.pi:
            phi += 2 * np.pi
        while phi >= np.pi:
            phi -= 2 * np.pi

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
            used = []
            for ip in range(i_phi - d_phi_bins, i_phi + d_phi_bins + 1):
                ip_wrapped = ip % self.n_phi  # φ wraps
                if ip_wrapped in used:
                    continue
                used.append(ip_wrapped)
                out.extend(self.tiles[ie][ip_wrapped])

        return out

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

@njit
def myFunc(x, y):
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
def compute_score(
    refPt, refEta, refPhi, refP,
    tsEta, tsPhi, tsEnergy, wEp=0.5
):
    rEp_norm = (tsEnergy - refP) / refP
    deltaPhi = refPhi - tsPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    deta = refEta - tsEta
    dR2 = deta ** 2 + deltaPhi ** 2
    pullR2 = dR2 / myFunc(refPt, refEta)**2
    score = wEp * rEp_norm**2 + (1-wEp) * pullR2
    return score**0.5

@njit
def normTracksters(x, y):
    #x, y = energy ,eta
    y = np.abs(y)
    a = 1.49662496e+00
    b = 4.04830636e+01
    c = -1.19840947e+01
    d = -4.04834456e+01
    e = 9.99984896e-01
    f = -1.67329993e-03
    g = 1.06828890e+01
    h = 1.09862164e+00

    if x>200: x=200
    val = a + b*x + c*y + d*x**e + g * y**h  + f*x*y
    return val #np.clip(val, 0.008, 1)

def fill_tiles_ec(eta, phi, tracksterTiles):
    for i in range(len(eta)):
        tracksterTiles.fill(eta[i], phi[i], i)
#         print("fill tile with ", eta[i], phi[i], idx[i])

def build_ts_edges(tsEta, tsPhi, tsZ, tsEnergy, tiles, dr_cut=0.05, shift=0.5, weight=4):
    edges = []
    dr_cut_2 = dr_cut**2

    for i in range(len(tsEta)):
        eta_i = tsEta[i]
        phi_i = tsPhi[i]
        en_i = tsEnergy[i]
        # get neighboring candidate tracksters
        d_eta_bins = min(int(np.ceil(dr_cut / tiles.getEtaWidth())), tiles.getNumEta())
        d_phi_bins = min(int(np.ceil(dr_cut / tiles.getPhiWidth())), tiles.getNumPhi())
        neigh = tiles.get_window(eta_i, phi_i, d_eta_bins=d_eta_bins, d_phi_bins=d_phi_bins)
        for j in neigh:
            # enforce outward direction
            if abs(tsZ[j]) <= abs(tsZ[i]):
                continue

            dr2 = distWrap2_numba(eta_i, phi_i, tsEta[j], tsPhi[j])

            if dr2 < dr_cut_2:
                en = max(en_i, tsEnergy[j])
                eta = eta_i if en_i>tsEnergy[j] else tsEta[j]
                edges.append((i, j, (dr2**0.5/normTracksters(en, eta)-shift)*weight))
    return edges

def build_trk_ts_edges(trkEta, trkPhi, trkPt, trkP, tsEta, tsPhi, tsEnergy, tiles, dr_cut=0.05, shift=1, weight=2):
    edges = []
    dr_cut_2 = dr_cut**2

    for i in range(len(trkEta)):
        eta_i = trkEta[i]
        phi_i = trkPhi[i]
        pt_i = trkPt[i]
        p_i = trkP[i]
        # get neighboring candidate tracksters
        d_eta_bins = min(int(np.ceil(dr_cut / tiles.getEtaWidth())), tiles.getNumEta())
        d_phi_bins = min(int(np.ceil(dr_cut / tiles.getPhiWidth())), tiles.getNumPhi())
        neigh = tiles.get_window(eta_i, phi_i, d_eta_bins=d_eta_bins, d_phi_bins=d_phi_bins)

        for j in neigh:
            dr2 = distWrap2(eta_i, phi_i, tsEta[j], tsPhi[j])

            if dr2 < dr_cut_2:
                score = compute_score(pt_i, eta_i, phi_i, p_i, tsEta[j], tsPhi[j], tsEnergy[j])
                edges.append((i, j, (score-shift)*weight))
    return edges


from collections import defaultdict
from ortools.graph.python import min_cost_flow

# does not allow tracksters sharing between candidates
def build_and_solve_flow_event_singleTs(tracksEv, tsLinksEv, endcap,
                                        dr_cut=0.02, neutral_penalty=1.0, ts_ts_score_shift=0.5, track_ts_score_shift=1,
                                        ts_ts_score_weight=1, track_ts_score_weight=1, n_eta = 30):
    # Filter tracks and tracksters
    if endcap == 0:
        mask3 = tracksEv.track_hgcal_eta < 0
        tsMask = tsLinksEv.barycenter_eta < 0
        tracksterTiles_ec = EtaPhiTiles(eta_min=-3.0, eta_max=-1.5, n_eta=n_eta)
    else:
        mask3 = tracksEv.track_hgcal_eta > 0
        tsMask = tsLinksEv.barycenter_eta > 0
        tracksterTiles_ec = EtaPhiTiles(eta_min= 1.5, eta_max= 3.0, n_eta=n_eta)

    mask1 = np.logical_and(tracksEv.track_hgcal_pt >= 1.0, tracksEv.track_p >= 2.0)
    mask2 = np.logical_and(np.abs(tracksEv.track_hgcal_eta) >= 1.5,
                           np.abs(tracksEv.track_hgcal_eta) <= 3.0)
    mask = mask1 & mask2 & mask3

    tracks_id = tracksEv.track_id[mask]
    tracks_eta = tracksEv.track_hgcal_eta[mask]
    tracks_phi = tracksEv.track_hgcal_phi[mask]
    tracks_pt = tracksEv.track_hgcal_pt[mask]
    tracks_p = tracksEv.track_p[mask]

    n_tracks = len(tracks_id)
    tsIdMap = np.arange(len(tsLinksEv.barycenter_eta))[tsMask]
    tsEta = tsLinksEv.barycenter_eta[tsMask]
    tsPhi = tsLinksEv.barycenter_phi[tsMask]
    tsZ = tsLinksEv.barycenter_z[tsMask]
    tsEnergy = tsLinksEv.raw_energy[tsMask]
    n_TS = len(tsEnergy)

    # tiles filling
    fill_tiles_ec(tsEta, tsPhi, tracksterTiles_ec)

    # Track→TS edges
    track_ts_edges = build_trk_ts_edges(tracks_eta, tracks_phi, tracks_pt, tracks_p, tsEta, tsPhi, tsEnergy, tracksterTiles_ec, dr_cut=dr_cut, shift=track_ts_score_shift, weight=track_ts_score_weight)

    # TS→TS edges
    ts_ts_edges = build_ts_edges(tsEta, tsPhi, tsZ, tsEnergy, tracksterTiles_ec, dr_cut=dr_cut, shift=ts_ts_score_shift, weight=ts_ts_score_weight)

    mcf = min_cost_flow.SimpleMinCostFlow()
    SRC = 0
    TRACK_OFFSET = 1
    TS_IN_OFFSET = TRACK_OFFSET + n_tracks
    TS_OUT_OFFSET = TS_IN_OFFSET + n_TS
    SNK = TS_OUT_OFFSET + n_TS
    N_total = SNK + 1

    node_id = np.zeros(N_total)
    node_id[SRC] = -1
    node_id[TRACK_OFFSET:TS_IN_OFFSET] = 0
    node_id[TS_IN_OFFSET:TS_OUT_OFFSET] = 1
    node_id[TS_OUT_OFFSET:SNK] = 2
    node_id[SNK] = 3

    # 3a. Source → Track
    for trk_idx in range(n_tracks):
        mcf.add_arc_with_capacity_and_unit_cost(SRC, TRACK_OFFSET + trk_idx, n_TS, -100)

    # 3a-bis. Source → TS (neutral start)
    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(SRC, TS_IN_OFFSET + ts_idx, 1, 0)

    # 3b. Track → TS
    for trk_idx, ts_idx, score in track_ts_edges:
        mcf.add_arc_with_capacity_and_unit_cost(TRACK_OFFSET + trk_idx, TS_IN_OFFSET + ts_idx, 1, int(score*1000))  # OR-Tools requires integer costs

    # 3c. TSin → TSout (same trackster for exclusivity)
    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(TS_IN_OFFSET + ts_idx, TS_OUT_OFFSET + ts_idx, 1, 0) # capacity = 1 → exclusivity

    # 3c-bis. TS → TS
    for i, j, score in ts_ts_edges:
        mcf.add_arc_with_capacity_and_unit_cost(TS_OUT_OFFSET + i, TS_IN_OFFSET + j, 1, int(score*1000))

    # 3d. TS → Sink (neutral)
    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(TS_OUT_OFFSET + ts_idx, SNK, 1, int(neutral_penalty*1000))

    # 3e. Track → Sink (track-only)
    for trk_idx in range(n_tracks):
        mcf.add_arc_with_capacity_and_unit_cost(TRACK_OFFSET + trk_idx, SNK, 1, int(neutral_penalty*1000))

    # 3f. Supplies
    mcf.set_node_supply(SRC, n_TS)
    mcf.set_node_supply(SNK, -n_TS)
    # Tracks and TS nodes = 0 (default)

    # --------------------
    # 4. Solve min-cost flow
    # --------------------
    def arc_type(u, v):
        if u == SRC:
            return "SRC→*"
        if TRACK_OFFSET <= u < TS_IN_OFFSET and TS_IN_OFFSET <= v < TS_OUT_OFFSET:
            return "TRK→TS"
        if TS_IN_OFFSET <= u < TS_OUT_OFFSET and TS_OUT_OFFSET <= v < SNK:
            return "INTERNAL TS"
        if TS_OUT_OFFSET <= u < SNK and TS_IN_OFFSET <= v < TS_OUT_OFFSET:
            return "TS→TS"
        if v == SNK:
            return "*→SNK"
        return "OTHER"

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError("Min-cost flow did not find an optimal solution")

    total_cost = 0
    for arc in range(mcf.num_arcs()):
        flow = mcf.flow(arc)
        if flow <= 0:
            continue
        cap = mcf.capacity(arc)
        unit_cost = mcf.unit_cost(arc)
        cost = flow * unit_cost

        u = mcf.tail(arc)
        v = mcf.head(arc)

        total_cost += cost

    # --------------------
    # 5. Decode flow into candidates
    # --------------------
    used_out = defaultdict(list)
    for arc in range(mcf.num_arcs()):
        if mcf.flow(arc) > 0:
            u = mcf.tail(arc)
            v = mcf.head(arc)
            used_out[u].append(v)

    charged_candidates = []
    for trk_node in used_out[SRC]:
        if not (TRACK_OFFSET <= trk_node < TS_IN_OFFSET):
            continue

        trk_idx = trk_node - TRACK_OFFSET
        ts_set = []
        # follow all outgoing branches from the track
        for v in used_out.get(trk_node, []):
            cur = v
            while cur != SNK:
                # TS_in
                if TS_IN_OFFSET <= cur < TS_OUT_OFFSET:
                    ts_set.append(cur - TS_IN_OFFSET)

                nexts = used_out.get(cur, [])
                if len(nexts) == 0 or len(nexts) > 1:
                    raise RuntimeError("Branching below TS_in should not happen and should be linked with SNK "+str(nexts))
                cur = nexts[0]

        charged_candidates.append({
            "track": int(tracks_id[trk_idx]), # [trk_idx, int(tracks_id[trk_idx])],
            "tracksters": [int(tsIdMap[ts_idx]) for ts_idx in ts_set] #(ts_idx,
        })

#     used_ts = [ts for c in charged_candidates for ts in c["tracksters"]]

    neutral_candidates = []
    for start in used_out[SRC]:
        if not (TS_IN_OFFSET <= start < TS_OUT_OFFSET):
            continue

        ts_idx = start - TS_IN_OFFSET

        cur = start
        ts_chain = []
        while cur != SNK:
            if TS_IN_OFFSET <= cur < TS_OUT_OFFSET:
                ts_chain.append(cur - TS_IN_OFFSET)

            nexts = used_out.get(cur, [])
            if len(nexts) == 0:
                print("why not SNK?")
                break

            cur = nexts[0]

        neutral_candidates.append({
            "track": None,
            "tracksters": [int(tsIdMap[ts_idx]) for ts_idx in ts_chain]
        })

    candidates = charged_candidates + neutral_candidates

    return candidates, total_cost

def compute_sim_score_mcf(arc_cost, true_links, tracks_id, tsIdMap, tsZ, SRC, SNK, TRACK_OFFSET, TS_IN_OFFSET):

    # Map track and TS to internal node indices
    track_map = {tid: i + TRACK_OFFSET for i, tid in enumerate(tracks_id)}
    ts_map = {ts: i + TS_IN_OFFSET for i, ts in enumerate(tsIdMap)}
    tsZ = np.abs(tsZ)
    total_cost = 0

    for tl in true_links:
        trk = tl["track_id"]
        tsList = tl["reco_ts"]
        ts_indices = np.array([ts_map[ts] for ts in tsList])
        n_TS = len(ts_indices)
        mcf = min_cost_flow.SimpleMinCostFlow()

        if trk is None or trk[0] not in track_map:
            for i, ts in enumerate(ts_indices):
                mcf.add_arc_with_capacity_and_unit_cost(SRC, ts, 1, arc_cost[(SRC, ts)])
                mcf.add_arc_with_capacity_and_unit_cost(ts, SNK, 1, arc_cost[(ts, SNK)])
                for other in ts_indices[ts_indices!=ts]:
                    if tsZ[other - TS_IN_OFFSET] > tsZ[ts - TS_IN_OFFSET]:
                        mcf.add_arc_with_capacity_and_unit_cost(ts, other, 1, arc_cost[(ts, other)])
        else:
            trk_idx = track_map[trk[0]]
            mcf.add_arc_with_capacity_and_unit_cost(SRC, trk_idx, n_TS, arc_cost[(SRC, trk_idx)])

            for ts in ts_indices:
                mcf.add_arc_with_capacity_and_unit_cost(trk_idx, ts, 1, arc_cost[(trk_idx, ts)])
                mcf.add_arc_with_capacity_and_unit_cost(ts, SNK, 1, arc_cost[(ts, SNK)])
                for other in ts_indices[ts_indices!=ts]:
                    if tsZ[other - TS_IN_OFFSET] > tsZ[ts - TS_IN_OFFSET]:
                        mcf.add_arc_with_capacity_and_unit_cost(ts, other, 1, arc_cost[(ts, other)])

        mcf.set_node_supply(SRC, n_TS)
        mcf.set_node_supply(SNK, -n_TS)
        status = mcf.solve()
        if status != mcf.OPTIMAL:
            raise RuntimeError("Min-cost flow did not find an optimal solution")
        for arc in range(mcf.num_arcs()):
            flow = mcf.flow(arc)
            if flow <= 0:
                continue
            total_cost += mcf.unit_cost(arc) * flow

    return total_cost

# sim ost
def build_flow_event_singleTs(tracksEv, tsLinksEv, true_links, endcap,
                              dr_cut=0.02, neutral_penalty=1.0, ts_ts_score_shift=0.5, track_ts_score_shift=1,
                              ts_ts_score_weight=1, track_ts_score_weight=1, n_eta = 30):
    # Filter tracks and tracksters
    if endcap == 0:
        mask3 = tracksEv.track_hgcal_eta < 0
        tsMask = tsLinksEv.barycenter_eta < 0
        tracksterTiles_ec = EtaPhiTiles(eta_min=-3.0, eta_max=-1.5, n_eta=n_eta)
    else:
        mask3 = tracksEv.track_hgcal_eta > 0
        tsMask = tsLinksEv.barycenter_eta > 0
        tracksterTiles_ec = EtaPhiTiles(eta_min= 1.5, eta_max= 3.0, n_eta=n_eta)

    mask1 = np.logical_and(tracksEv.track_hgcal_pt >= 1.0, tracksEv.track_p >= 2.0)
    mask2 = np.logical_and(np.abs(tracksEv.track_hgcal_eta) >= 1.5,
                           np.abs(tracksEv.track_hgcal_eta) <= 3.0)
    mask = mask1 & mask2 & mask3

    tracks_id = tracksEv.track_id[mask]
    tracks_eta = tracksEv.track_hgcal_eta[mask]
    tracks_phi = tracksEv.track_hgcal_phi[mask]
    tracks_pt = tracksEv.track_hgcal_pt[mask]
    tracks_p = tracksEv.track_p[mask]

    n_tracks = len(tracks_id)
    tsIdMap = np.arange(len(tsLinksEv.barycenter_eta))[tsMask]
    tsEta = tsLinksEv.barycenter_eta[tsMask]
    tsPhi = tsLinksEv.barycenter_phi[tsMask]
    tsZ = tsLinksEv.barycenter_z[tsMask]
    tsEnergy = tsLinksEv.raw_energy[tsMask]
    n_TS = len(tsEnergy)

    # tiles filling
    fill_tiles_ec(tsEta, tsPhi, tracksterTiles_ec)

    track_ts_edges = build_trk_ts_edges(tracks_eta, tracks_phi, tracks_pt, tracks_p, tsEta, tsPhi, tsEnergy, tracksterTiles_ec, dr_cut=dr_cut, shift=track_ts_score_shift, weight=track_ts_score_weight)
    ts_ts_edges = build_ts_edges(tsEta, tsPhi, tsZ, tsEnergy, tracksterTiles_ec, dr_cut=dr_cut, shift=ts_ts_score_shift, weight=ts_ts_score_weight)

    mcf = min_cost_flow.SimpleMinCostFlow()
    SRC = 0
    TRACK_OFFSET = 1
    TS_IN_OFFSET = TRACK_OFFSET + n_tracks
    TS_OUT_OFFSET = TS_IN_OFFSET + n_TS
    SNK = TS_OUT_OFFSET + n_TS
    N_total = SNK + 1

    node_id = np.zeros(N_total)
    node_id[SRC] = -1
    node_id[TRACK_OFFSET:TS_IN_OFFSET] = 0
    node_id[TS_IN_OFFSET:TS_OUT_OFFSET] = 1
    node_id[TS_OUT_OFFSET:SNK] = 2
    node_id[SNK] = 3

    for trk_idx in range(n_tracks):
        mcf.add_arc_with_capacity_and_unit_cost(SRC, TRACK_OFFSET + trk_idx, n_TS, -100)

    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(SRC, TS_IN_OFFSET + ts_idx, 1, 0)

    for trk_idx, ts_idx, score in track_ts_edges:
        mcf.add_arc_with_capacity_and_unit_cost(TRACK_OFFSET + trk_idx, TS_IN_OFFSET + ts_idx, 1, int(score*1000))  # OR-Tools requires integer costs

    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(TS_IN_OFFSET + ts_idx, TS_OUT_OFFSET + ts_idx, 1, 0) # capacity = 1 → exclusivity

    for i, j, score in ts_ts_edges:
        mcf.add_arc_with_capacity_and_unit_cost(TS_OUT_OFFSET + i, TS_IN_OFFSET + j, 1, int(score*1000))

    for ts_idx in range(n_TS):
        mcf.add_arc_with_capacity_and_unit_cost(TS_OUT_OFFSET + ts_idx, SNK, 1, int(neutral_penalty*1000))

    for trk_idx in range(n_tracks):
        mcf.add_arc_with_capacity_and_unit_cost(TRACK_OFFSET + trk_idx, SNK, 1, int(neutral_penalty*1000))

    mcf.set_node_supply(SRC, n_TS)
    mcf.set_node_supply(SNK, -n_TS)

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError("Min-cost flow did not find an optimal solution")

    def arc_type(u, v):
        if u == SRC:
            return "SRC→*"
        if TRACK_OFFSET <= u < TS_IN_OFFSET and TS_IN_OFFSET <= v < TS_OUT_OFFSET:
            return "TRK→TS"
        if TS_IN_OFFSET <= u < TS_OUT_OFFSET and TS_OUT_OFFSET <= v < SNK:
            return "INTERNAL TS"
        if TS_OUT_OFFSET <= u < SNK and TS_IN_OFFSET <= v < TS_OUT_OFFSET:
            return "TS→TS"
        if v == SNK:
            return "*→SNK"
        return "OTHER"

    arc_cost = {}
#     print(" ArcType   u -> v   Flow / Cap   Unit Cost   Cost")
    for arc in range(mcf.num_arcs()):
        flow = mcf.flow(arc)
        cap = mcf.capacity(arc)
        unit_cost = mcf.unit_cost(arc)
        cost = unit_cost * flow

        u = mcf.tail(arc)
        v = mcf.head(arc)
        typeArc = arc_type(u,v)
        if typeArc == "TS→TS":
            u = u - n_TS
        if typeArc == "*→SNK" and u >= TS_OUT_OFFSET:
            u = u - n_TS
#         print(
#             f"{typeArc:8s} "
#             f"{u:4d}->{v:4d}   "
#             f"{flow:3d}/{cap:3d}   "
#             f"{unit_cost:7d}   "
#             f"{cost:7d}"
#         )
        arc_cost[(u, v)] = unit_cost

    if np.any(mask) or np.any(tsMask):
        total_sim_cost = compute_sim_score_mcf(arc_cost, true_links, tracks_id, tsIdMap, tsZ, SRC, SNK, TRACK_OFFSET, TS_IN_OFFSET)
        return total_sim_cost
    else:
	    return 0

def compute_efficiency(events, simtrackstersSC, tracks, all_candidates, tracksterLinks, associations):
    eff_flags = []
    eta_vals = []
    ene_vals = []
    for ev in events: #range(len(simtrackstersSC)): FIXME

        stsSCEv   = simtrackstersSC[ev]
        tracksEv  = tracks[ev]
        tsLinksEv = tracksterLinks[ev]
        assEv     = associations[ev]

        # Build reco link lookup
        links_ev = all_candidates[ev] # FIXME!! [2*ev] + all_candidates[2*ev+1]

        track_to_cp = {}
        for i in range(len(stsSCEv.trackIdx)):
            if len(stsSCEv.trackIdx[i]) == 0 or stsSCEv.trackIdx[i][0]==-1:
                continue
            track_to_cp[stsSCEv.trackIdx[i][0]] = stsSCEv.CPidx[i]

        ts_to_track = {}
        for l in links_ev:
            trk = l['track']
            if trk is None:
                continue
            for ts in l['tracksters']:
                ts_to_track[ts] = trk

        for idx in range(len(stsSCEv.trackIdx)):

            trk_ids = stsSCEv.trackIdx[idx]
            if len(trk_ids) == 0:
                continue

            itrk = int(trk_ids[0]) # assume 1 track
            trk_pos = find_track_id(tracksEv.track_id, itrk)
            if trk_pos == -1:
                continue

            pt  = tracksEv.track_hgcal_pt[trk_pos]
            p   = tracksEv.track_p[trk_pos]

            if pt < 1 or p < 2:
                continue

            maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.999
            assocRecoTsIds = assEv.ticlTracksterLinks_simToReco_SC[idx]
            tsEnergy = tsLinksEv.raw_energy[assocRecoTsIds]
            sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
            maskEnergy = sharedEnergy / tsEnergy > 0.5

            true_ts = assocRecoTsIds[maskScore & maskEnergy]
            if len(true_ts) == 0:
                continue

            matched = False
            cp_idx = track_to_cp.get(itrk)
            for ts in true_ts:
                if ts not in ts_to_track: continue
                if ts_to_track[ts]==itrk:
                    # efficiente
                    eff_flags.append(True)
                    matched=True
                    break
                elif track_to_cp.get(ts_to_track[ts]) == cp_idx:
                    eff_flags.append(True)
                    matched=True
                    break

            if not matched:
                eff_flags.append(False)

            eta_vals.append(stsSCEv.barycenter_eta[idx])
            ene_vals.append(stsSCEv.regressed_energy[idx])


    eff_flags = arr(eff_flags)
    eta_vals = np.abs(arr(eta_vals))
    ene_vals = arr(ene_vals)

    return eff_flags, ene_vals, eta_vals

def compute_fake(events, simtrackstersSC, tracks, all_candidates, tracksterLinks, associations):
    ene_fake = []
    eta_fake = []
    fake = []

    for  ev in events: #range(len(simtrackstersSC)): FIXME
        tsLinks = tracksterLinks[ev]
        tracksEv = tracks[ev]
        stsSCEv = simtrackstersSC[ev]
        assEv = associations[ev]

        track_to_cp = {}
        for i in range(len(stsSCEv.trackIdx)):
            if len(stsSCEv.trackIdx[i]) == 0 or stsSCEv.trackIdx[i][0]==-1:
                continue
            track_to_cp[stsSCEv.trackIdx[i][0]] = stsSCEv.CPidx[i]

        # loop over links_ev
        links_ev = all_candidates[ev] # FIXME!!! [2*ev] + all_candidates[2*ev+1]
        for link in links_ev:
            ts_in_cand = link['tracksters']
            tk_in_cand = link['track']
            if len(ts_in_cand)==0: continue
            trkSim = []
            for recoTs in ts_in_cand:
                recoAssScore = assEv.ticlTracksterLinks_recoToSim_SC_score[recoTs]
                recoAssEne = assEv.ticlTracksterLinks_recoToSim_SC_sharedE[recoTs]
                recoAssIdx = assEv.ticlTracksterLinks_recoToSim_SC[recoTs]
                good = np.logical_and(recoAssScore<0.9 , recoAssEne / stsSCEv.raw_energy[recoAssIdx] > 0.5)
                if not np.any(good): continue
                recoAssScore = recoAssScore[good][0]
                recoAssIdx = recoAssIdx[good][0]
                recoAssEne = recoAssEne[good][0]
                if len(stsSCEv.trackIdx[recoAssIdx]):
                    trkSim.append(stsSCEv.trackIdx[recoAssIdx][0])

            if len(trkSim)==0 and tk_in_cand==None: # true neutral
                fake.append(False)
            elif len(trkSim)==0 and tk_in_cand!=None or len(trkSim)>0 and tk_in_cand==None:
                # false charged / neutral
                fake.append(True)
            elif tk_in_cand!=None and len(trkSim)>0 :
                if tk_in_cand not in trkSim: # true charged but wrong link
                    true_cp = [track_to_cp[ts] for ts in trkSim]
                    if tk_in_cand in track_to_cp and track_to_cp[tk_in_cand] in true_cp: # but same cp so ok
                        fake.append(False)
                    else: # wrong cp
                        fake.append(True)
                else: # true charged and true link
                    fake.append(False)
            else:
                print("I missed sth", tk_in_cand, trkSim)

            ene_fake.append(sum(tsLinks.raw_energy[ts_in_cand]))
            eta_fake.append(np.mean(tsLinks.barycenter_eta[ts_in_cand]))
    eta_fake = np.abs(arr(eta_fake))
    ene_fake = arr(ene_fake)
    fake = arr(fake)

    return fake, ene_fake, eta_fake

def compute_average(metric, variable, CUT=50):
    mask = variable<CUT
    if np.sum(mask)==0 or np.sum(mask)==len(variable):
        low = np.mean(metric)
        high = np.mean(metric)
    else:
        low = np.mean(metric[variable<CUT])
        high = np.mean(metric[variable>=CUT])
    return low, high

'''
FIXME set two endcaps with this sample
allFiles_simtrackstersSC = []
#allFiles_associations = []
allFiles_tracks = []
#allFiles_tracksterLinks = []

for PT in [10, 50, 100, 200]:
        for ETA in [1.7, 2.2, 2.7]:
            label = "pt"+str(PT)+"_eta"+str(ETA).replace(".","p")
            file = uproot.open("/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_"+label+"/histo_"+label+".root")

            allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
            allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
            alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
            allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

            simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys, entry_stop=100)
            associations = allassociations.arrays(assKeys, entry_stop=100)
            tracks = alltracks.arrays(tracksKeys, entry_stop=100)
            tracksterLinks = allticlTracksterLinks.arrays(tsKeys, entry_stop=100)

            allFiles_simtrackstersSC.append(simtrackstersSC)
            #allFiles_associations.append(associations)
            allFiles_tracks.append(tracks)
            #allFiles_tracksterLinks.append(tracksterLinks)
'''

file = uproot.open("/eos/user/a/aperego/Timing/root_files/multiParticleInConedR0.root")

allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys, entry_stop=1000)
tracks = alltracks.arrays(tracksKeys, entry_stop=1000)
tracksterLinks = allticlTracksterLinks.arrays(tsKeys, entry_stop=1000)
associations = allassociations.arrays(assKeys, entry_stop=1000)

async def main_event_run(x):
    eff = []
    ene_eff = []
    eta_eff = []
    fake = []
    ene_fake = []
    eta_fake = []
    cost_diff = []
    for i in range(1): #len(allFiles_simtrackstersSC)):
        '''
        simtrackstersSC = allFiles_simtrackstersSC[i]
        #associations = allFiles_associations[i]
        tracks = allFiles_tracks[i]
        #tracksterLinks = allFiles_tracksterLinks[i]
        '''

        tot_events = 100 #len(tracks)
        all_candidates = []
        for ev in np.random.randint(0,1000,100): #range(tot_events):
            tracksEv = tracks[ev]
            tsLinksEv = tracksterLinks[ev]
            assEv = associations[ev]
            stsSCEv = simtrackstersSC[ev]

            true_links_ev = []
            for idx in range(len(stsSCEv.trackIdx)):
                # charged / neutral identification
                trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
                is_charged = trk_id != -1

                # select good reco tracksters
                maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.99 # FIXME tracksters with score 0.997 but shEnergy very high
                assocRecoTsIds = assEv.ticlTracksterLinks_simToReco_SC[idx]
                tsEnergy = tsLinksEv.raw_energy[assocRecoTsIds]
                sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
                maskEnergy = sharedEnergy / tsEnergy > 0.5
                maskIdx = maskScore & maskEnergy
                assIndices = assocRecoTsIds[maskIdx]

                if len(assIndices)==0: continue

                sortTs = np.argsort(np.abs(tsLinksEv.barycenter_z[assIndices]))
                assIndices = assIndices[sortTs]

                true_links_ev.append({
                    "cp_id": stsSCEv.CPidx[idx],
                    "track_id": [int(trkIdx) for trkIdx in stsSCEv.trackIdx[idx]] if is_charged else None,
                    "reco_ts": [int(assIdx) for assIdx in assIndices]
                })

            for endcap in [1]: #FIXME !!!!! (0, 1):
                candidates_ec, reco_cost = build_and_solve_flow_event_singleTs(tracksEv, tsLinksEv, endcap=endcap, dr_cut=x[0], ts_ts_score_shift=x[1], track_ts_score_shift=x[2], ts_ts_score_weight=x[3], track_ts_score_weight=x[4], neutral_penalty=x[5])
                all_candidates.append(candidates_ec)
                sim_cost = build_flow_event_singleTs(tracksEv, tsLinksEv, true_links_ev, endcap=endcap, dr_cut=100, ts_ts_score_shift=x[1], track_ts_score_shift=x[2], ts_ts_score_weight=x[3], track_ts_score_weight=x[4], neutral_penalty=x[5])
                cost_diff.append((reco_cost-sim_cost)/sim_cost)

        eff_bin, ene_eff_bin, eta_eff_bin = compute_efficiency(simtrackstersSC, tracks, all_candidates, tracksterLinks, associations)
        fake_bin, ene_fake_bin, eta_fake_bin = compute_fake(simtrackstersSC, tracks, all_candidates, tracksterLinks, associations)

        eff.extend(eff_bin)
        ene_eff.extend(ene_eff_bin)
        eta_eff.extend(eta_eff_bin)
        fake.extend(fake_bin)
        ene_fake.extend(ene_fake_bin)
        eta_fake.extend(eta_fake_bin)


    eff = arr(eff)
    ene_eff = arr(ene_eff)
    fake = arr(fake)
    ene_fake = arr(ene_fake)
    #print(eff, fake, ene_eff, ene_fake)
    eff_low, eff_high = compute_average(eff, ene_eff)
    fake_low, fake_high = compute_average(fake, ene_fake)
    avg_cost = np.mean(cost_diff)

    #print(eff_low, eff_high, fake_low, fake_high)
    return [eff_low, eff_high, fake_low, fake_high, avg_cost]

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def process_event(ev, x, tracks, tracksterLinks, associations, simtrackstersSC):
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    assEv = associations[ev]
    stsSCEv = simtrackstersSC[ev]

    true_links_ev = []

    for idx in range(len(stsSCEv.trackIdx)):
        trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
        is_charged = trk_id != -1

        maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.99
        assocRecoTsIds = assEv.ticlTracksterLinks_simToReco_SC[idx]
        tsEnergy = tsLinksEv.raw_energy[assocRecoTsIds]
        sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]

        maskEnergy = sharedEnergy / tsEnergy > 0.5
        maskIdx = maskScore & maskEnergy
        assIndices = assocRecoTsIds[maskIdx]

        if len(assIndices) == 0:
            continue

        sortTs = np.argsort(np.abs(tsLinksEv.barycenter_z[assIndices]))
        assIndices = assIndices[sortTs]

        true_links_ev.append({
            "cp_id": stsSCEv.CPidx[idx],
            "track_id": [int(trkIdx) for trkIdx in stsSCEv.trackIdx[idx]] if is_charged else None,
            "reco_ts": [int(assIdx) for assIdx in assIndices]
        })
    for endcap in [1]: #FIXME !!!!! (0, 1):
        candidates_ec, reco_cost = build_and_solve_flow_event_singleTs(tracksEv, tsLinksEv, endcap=endcap, dr_cut=x[0], ts_ts_score_shift=x[1], track_ts_score_shift=x[2], ts_ts_score_weight=x[3], track_ts_score_weight=x[4], neutral_penalty=x[5])
        sim_cost = build_flow_event_singleTs(tracksEv, tsLinksEv, true_links_ev, endcap=endcap, dr_cut=100, ts_ts_score_shift=x[1], track_ts_score_shift=x[2], ts_ts_score_weight=x[3], track_ts_score_weight=x[4], neutral_penalty=x[5])

    cost_diff = (reco_cost - sim_cost) / sim_cost

    return {ev: candidates_ec}, cost_diff

def main_parallel_event_run(x):
    cost_diff = []
    all_candidates = {}

    events = np.random.randint(0, 1000, 100)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda ev: process_event(ev, x, tracks, tracksterLinks, associations, simtrackstersSC),
            events
        ))

    for candidates_ec, cost in results:
        all_candidates.update(candidates_ec)
        cost_diff.append(cost)

    eff, ene_eff, eta_eff = compute_efficiency(
        events, simtrackstersSC, tracks, all_candidates, tracksterLinks, associations
    )

    fake, ene_fake, eta_fake = compute_fake(
        events, simtrackstersSC, tracks, all_candidates, tracksterLinks, associations
    )

    eff = arr(eff)
    ene_eff = arr(ene_eff)
    fake = arr(fake)
    ene_fake = arr(ene_fake)
    eff_low, eff_high = compute_average(eff, ene_eff)
    fake_low, fake_high = compute_average(fake, ene_fake)
    avg_cost = np.mean(cost_diff)

    #print(eff_low, eff_high, fake_low, fake_high, avg_cost)
    return [eff_low, eff_high, fake_low, fake_high, avg_cost]

#def_params = [0.2, 0.5, 1, 40, 20, 1]
#print("default set " , def_params, " gives ", main_event_run(def_params))

import patatune

patatune.Logger.setLevel('DEBUG')

patatune.FileManager.saving_enabled = True
patatune.FileManager.headers_enabled = True
patatune.FileManager.saving_csv_enabled = True
patatune.FileManager.saving_pickle_enabled = False
patatune.FileManager.working_dir = "sim"

# dr, ts shift, trk shift, ts scale, trk scale, neutral
lb = [0., 0., 0., 0.1, 0.1, 0.01]
ub = [0.5, 5., 5., 70., 70., 70.]

objective = patatune.ElementWiseObjective(main_parallel_event_run, num_objectives=5, directions=['maximize', 'maximize', 'minimize', 'minimize', 'minimize'], objective_names=['eff_low', 'eff_high', 'fake_low', 'fake_high', 'costWRTsim'])

mopso = patatune.MOPSO( objective,
                        lower_bounds=lb, upper_bounds=ub,
                        param_names = ['dRmax', 'ts_shift', 'trk_shift', 'ts_weight', 'trk_weight', 'neutral'],
                        num_particles=100,
                        initial_particles_position="gaussian", default_point=[0.058315,  0.865011,  2.253941, 40.047265, 66.395847, 43.285349],
                        inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2)

pareto = mopso.optimize(num_iterations = 100)

