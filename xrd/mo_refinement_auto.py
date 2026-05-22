"""
Automated Rietveld refinement for Mo/Mo2C/MoC phase quantification.

Logic mirrors manual process:
1. Try each phase alone — pick the one with lowest Rwp as the "dominant" phase
2. Add remaining phases one at a time — keep if Rwp drops > RWP_IMPROVEMENT_THRESHOLD
3. For MoC: try all non-stoichiometric variants + stoichiometric — keep best Rwp
4. Output CSV table matching manual results format

Usage:
    python mo_refinement_auto.py
"""

import sys, os, glob, csv, warnings
import numpy as np

# ── GSAS-II paths ────────────────────────────────────────────────────────────
GSAS_PATHS = [
    '/Users/shane/g2full/GSAS-II',
    '/Users/shane/g2full/GSAS-II/GSASII',
]
for p in GSAS_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Toolkit path ─────────────────────────────────────────────────────────────
TOOLKIT_ROOT = '/Users/shane/Desktop/Porosoff Research/Repos/catalysis-toolkit'
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)

import GSASIIscriptable as G2sc

# ── USER CONFIG ──────────────────────────────────────────────────────────────
PATTERN_DIR   = '/Users/shane/Desktop/Eva_proj/XRD Index 3'
CIF_DIR       = '/Users/shane/Downloads/Reference cif'
INSTPRM_FILE  = '/Users/shane/Desktop/Porosoff Research/Repos/catalysis-toolkit/smartlab_Si640g.instprm'
OUTPUT_CSV    = '/Users/shane/Desktop/Eva_proj/refinement_results.csv'
WORK_DIR      = '/Users/shane/Desktop/Eva_proj/gsas_tmp'

# Rwp improvement threshold to keep an added phase (absolute %)
RWP_IMPROVEMENT_THRESHOLD = 2.0

# Phase CIF definitions
PHASE_CIFS = {
    'Mo':   os.path.join(CIF_DIR, 'Mo_Im3m_fixed.cif'),
    'Mo2C': os.path.join(CIF_DIR, 'Mo2C_Pbcn_fixed.cif'),
}

# MoC variants — stoichiometric last, try all, keep best Rwp
MOC_VARIANTS = {
    'MoC_0.66': os.path.join(CIF_DIR, 'MoC_0.66.cif'),
    'MoC_0.68': os.path.join(CIF_DIR, 'MoC_0.68.cif'),
    'MoC_0.70': os.path.join(CIF_DIR, 'MoC_0.70.cif'),
    'MoC_0.72': os.path.join(CIF_DIR, 'MoC_0.72.cif'),
    'MoC_0.74': os.path.join(CIF_DIR, 'MoC_0.74.cif'),
    'MoC_0.75': os.path.join(CIF_DIR, 'MoC_0.75.cif'),
    'MoC_1.00': os.path.join(CIF_DIR, 'MoC_cubic_fixed.cif'),
}

os.makedirs(WORK_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def write_instprm(work_dir, wavelength=1.5406):
    """Write a minimal Cu Ka instrument parameter file."""
    path = os.path.join(work_dir, 'instrument.instprm')
    # Use user instprm if it exists
    if os.path.isfile(INSTPRM_FILE):
        import shutil
        shutil.copy2(INSTPRM_FILE, path)
        return path
    # Fallback: write minimal Cu Ka
    lines = [
        '#GSAS-II instrument parameter file; do not add/delete items!',
        'Type:PXC',
        f'Lam1:{1.540593:.6f}',
        f'Lam2:{1.544414:.6f}',
        'I(L2)/I(L1):0.5000',
        'Zero:0.0',
        'Polariz.:0.7',
        'U:2.0', 'V:-2.0', 'W:5.0',
        'X:0.0', 'Y:0.0', 'Z:0.0',
        'SH/L:0.002',
        'Azimuth:0.0',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def load_csv_pattern(csv_path):
    """Load XRD pattern from CSV, return (tt, y_obs) arrays."""
    data = []
    with open(csv_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip().replace(',', ' ')
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
                if np.isfinite(x) and np.isfinite(y):
                    data.append((x, y))
            except ValueError:
                continue
    if not data:
        return None, None
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


def write_xye(path, tt, y_obs):
    """Write XYE file for GSAS-II."""
    sigma = np.sqrt(np.maximum(y_obs, 1.0))
    with open(path, 'w') as f:
        for x, y, s in zip(tt, y_obs, sigma):
            f.write(f'{x:.6f} {y:.6f} {s:.6f}\n')


def run_refinement(pattern_path, phase_dict, work_subdir):
    """
    Run a single Rietveld refinement with given phases.

    phase_dict: {'phasename': '/path/to/cif', ...}

    Returns dict with Rwp, GOF, chi2, and per-phase wt% and sigma.
    Returns None on failure.
    """
    os.makedirs(work_subdir, exist_ok=True)
    gpx_path  = os.path.join(work_subdir, 'refine.gpx')
    xye_path  = os.path.join(work_subdir, 'data.xye')
    inst_path = write_instprm(work_subdir)

    # Load and write pattern
    tt, y_obs = load_csv_pattern(pattern_path)
    if tt is None:
        return None
    write_xye(xye_path, tt, y_obs)

    try:
        gpx = G2sc.G2Project(newgpx=gpx_path)
        hist = gpx.add_powder_histogram(xye_path, inst_path)

        # Fix histogram scale
        try:
            hist.data['Sample Parameters']['Scale'] = [1.0, False]
        except Exception:
            pass

        # Add phases
        phase_objs = {}
        for pname, cif_path in phase_dict.items():
            try:
                phase_objs[pname] = gpx.add_phase(
                    cif_path, phasename=pname, histograms=[hist], fmthint='CIF')
            except Exception as e:
                print(f'    Phase {pname} failed to load: {e}')
                return None

        # Equal initial scales
        for pobj in phase_objs.values():
            hap = list(pobj.data['Histograms'].values())[0]
            hap['Scale'] = [0.1, True]

        # Background setup
        bg = hist.data['Background']
        bg[0] = ['chebyschev-1', True, 3,
                 float(np.percentile(y_obs, 5)), 0.0, 0.0]

        # Step 1: background only
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
        }, 'cycles': 5}])

        # Step 2: add zero shift
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
            'Instrument Parameters': ['Zero'],
        }, 'cycles': 5}])

        # Step 3: phase fractions
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
            'Instrument Parameters': ['Zero'],
        }, 'cycles': 10}])

        # Extract Rwp
        try:
            gsas_stats = hist.get_statistics()
            rwp = float(gsas_stats.get('Rwp', 99.0))
            gof = float(gsas_stats.get('GOF', 99.0))
        except Exception:
            res = hist.residuals
            rwp = float(res.get('wR', 99.0))
            gof = 99.0

        # Extract weight fractions from .lst file
        lst_path = gpx_path.replace('.gpx', '.lst')
        wt_fracs = {}
        wt_sigmas = {}
        if os.path.isfile(lst_path):
            with open(lst_path, 'r', errors='ignore') as f:
                content = f.read()
            import re
            # Parse: "Weight fraction : 0.xxx, sig 0.xxx"
            for match in re.finditer(
                    r'Phase fraction\s*:\s*([\d.]+),\s*sig\s*([\d.]+)\s*'
                    r'Weight fraction\s*:\s*([\d.]+),\s*sig\s*([\d.]+)\s*'
                    r'Phase:\s*([^\n]+)',
                    content, re.MULTILINE):
                pf, pf_sig, wf, wf_sig, pname = match.groups()
                pname = pname.strip().split(' in ')[0].strip()
                wt_fracs[pname]  = float(wf) * 100
                wt_sigmas[pname] = float(wf_sig) * 100

        # Fallback: compute from scale factors if lst parsing failed
        if not wt_fracs:
            raw_scales = {}
            for pname, pobj in phase_objs.items():
                hap = list(pobj.data['Histograms'].values())[0]
                raw_scales[pname] = hap.get('Scale', [0])[0]
            total = sum(raw_scales.values()) or 1e-10
            for pname, s in raw_scales.items():
                wt_fracs[pname]  = (s / total) * 100
                wt_sigmas[pname] = None

        # Chi2
        try:
            chi2 = gof ** 2
        except Exception:
            chi2 = None

        return {
            'Rwp':      round(rwp, 3),
            'GOF':      round(gof, 3),
            'chi2':     round(chi2, 3) if chi2 else None,
            'wt_fracs': wt_fracs,
            'wt_sigmas': wt_sigmas,
            'phases':   list(phase_dict.keys()),
        }

    except Exception as e:
        print(f'    Refinement failed: {e}')
        return None
    finally:
        import shutil
        shutil.rmtree(work_subdir, ignore_errors=True)


def select_phases_for_pattern(pattern_path, pattern_name):
    """
    Sequential phase selection matching manual process:
    1. Try each of Mo, Mo2C, MoC variants alone — pick lowest Rwp
    2. Add remaining phases one at a time — keep if Rwp drops > threshold
    3. For MoC: try all variants, keep best
    """
    print(f'\n{"="*60}')
    print(f'Pattern: {pattern_name}')
    print(f'{"="*60}')

    base_dir = os.path.join(WORK_DIR, pattern_name)

    # ── Step 1: find dominant phase ──────────────────────────────────
    print('Step 1: Finding dominant phase...')
    best_single = {'Rwp': 999, 'name': None, 'cif': None, 'result': None}

    # Try Mo and Mo2C
    for pname, cif in PHASE_CIFS.items():
        print(f'  Trying {pname} alone...')
        result = run_refinement(
            pattern_path, {pname: cif},
            os.path.join(base_dir, f'single_{pname}'))
        if result:
            print(f'    Rwp = {result["Rwp"]:.3f}%')
            if result['Rwp'] < best_single['Rwp']:
                best_single = {'Rwp': result['Rwp'], 'name': pname,
                               'cif': cif, 'result': result}

    # Try MoC variants
    best_moc = {'Rwp': 999, 'name': None, 'cif': None}
    for moc_name, moc_cif in MOC_VARIANTS.items():
        print(f'  Trying {moc_name} alone...')
        result = run_refinement(
            pattern_path, {moc_name: moc_cif},
            os.path.join(base_dir, f'single_{moc_name}'))
        if result:
            print(f'    Rwp = {result["Rwp"]:.3f}%')
            if result['Rwp'] < best_moc['Rwp']:
                best_moc = {'Rwp': result['Rwp'], 'name': moc_name,
                            'cif': moc_cif}
            if result['Rwp'] < best_single['Rwp']:
                best_single = {'Rwp': result['Rwp'], 'name': moc_name,
                               'cif': moc_cif, 'result': result}

    print(f'\n  Dominant phase: {best_single["name"]} '
          f'(Rwp={best_single["Rwp"]:.3f}%)')

    # ── Step 2: build active phase set ───────────────────────────────
    active_phases = {best_single['name']: best_single['cif']}
    current_rwp   = best_single['Rwp']

    # Determine remaining non-MoC phases to try
    remaining = {k: v for k, v in PHASE_CIFS.items()
                 if k != best_single['name']}

    # Add non-MoC phases
    for pname, cif in remaining.items():
        trial = dict(active_phases)
        trial[pname] = cif
        print(f'\nStep 2: Trying adding {pname}...')
        result = run_refinement(
            pattern_path, trial,
            os.path.join(base_dir, f'add_{pname}'))
        if result:
            print(f'  Rwp: {current_rwp:.3f}% → {result["Rwp"]:.3f}%  '
                  f'(Δ={current_rwp - result["Rwp"]:.3f}%)')
            if current_rwp - result['Rwp'] > RWP_IMPROVEMENT_THRESHOLD:
                print(f'  ✓ Keeping {pname}')
                active_phases[pname] = cif
                current_rwp = result['Rwp']
            else:
                print(f'  ✗ Dropping {pname} (no improvement)')

    # Add MoC — try all variants if MoC not already dominant
    moc_in_active = any('MoC' in k for k in active_phases)
    if not moc_in_active:
        print(f'\nStep 2: Trying MoC variants...')
        best_moc_trial = {'Rwp': current_rwp, 'name': None,
                          'cif': None, 'result': None}
        for moc_name, moc_cif in MOC_VARIANTS.items():
            trial = dict(active_phases)
            trial[moc_name] = moc_cif
            result = run_refinement(
                pattern_path, trial,
                os.path.join(base_dir, f'add_{moc_name}'))
            if result:
                print(f'  {moc_name}: Rwp={result["Rwp"]:.3f}%')
                if result['Rwp'] < best_moc_trial['Rwp']:
                    best_moc_trial = {'Rwp': result['Rwp'],
                                      'name': moc_name,
                                      'cif': moc_cif,
                                      'result': result}

        if (best_moc_trial['name'] and
                current_rwp - best_moc_trial['Rwp'] > RWP_IMPROVEMENT_THRESHOLD):
            print(f'  ✓ Keeping {best_moc_trial["name"]} '
                  f'(Rwp={best_moc_trial["Rwp"]:.3f}%)')
            active_phases[best_moc_trial['name']] = best_moc_trial['cif']
            current_rwp = best_moc_trial['Rwp']
        else:
            print(f'  ✗ No MoC variant improved Rwp enough')

    # ── Step 3: final refinement with selected phases ─────────────────
    print(f'\nFinal phases: {list(active_phases.keys())}')
    final_result = run_refinement(
        pattern_path, active_phases,
        os.path.join(base_dir, 'final'))

    if final_result:
        final_result['active_phases'] = list(active_phases.keys())
        final_result['moc_variant'] = next(
            (k for k in active_phases if 'MoC' in k), None)
    return final_result


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    # Find all pattern files
    pattern_files = sorted(glob.glob(os.path.join(PATTERN_DIR, 'M*.csv')))
    if not pattern_files:
        print(f'No pattern files found in {PATTERN_DIR}')
        return

    print(f'Found {len(pattern_files)} patterns: '
          f'{[os.path.basename(p) for p in pattern_files]}')

    rows = []
    for pattern_path in pattern_files:
        pattern_name = os.path.splitext(os.path.basename(pattern_path))[0]
        result = select_phases_for_pattern(pattern_path, pattern_name)

        if result is None:
            print(f'  FAILED: {pattern_name}')
            rows.append({'Index': pattern_name, 'Error': 'refinement failed'})
            continue

        # Build output row
        wf = result['wt_fracs']
        ws = result['wt_sigmas']

        def get_wf(keys):
            for k in keys:
                if k in wf:
                    return round(wf[k], 1)
            return 0

        def get_ws(keys):
            for k in keys:
                if k in ws and ws[k] is not None:
                    return round(ws[k], 3)
            return None

        mo_wf   = get_wf(['Mo'])
        mo_ws   = get_ws(['Mo'])
        mo2c_wf = get_wf(['Mo2C'])
        mo2c_ws = get_ws(['Mo2C'])
        moc_wf  = get_wf([k for k in wf if 'MoC' in k])
        moc_ws  = get_ws([k for k in ws if 'MoC' in k])
        moc_var = result.get('moc_variant', 'none')

        row = {
            'Index':            pattern_name,
            'Mo (Im3m) wt%':    mo_wf,
            'Mo sigma':         mo_ws,
            'Mo2C (ortho) wt%': mo2c_wf,
            'Mo2C sigma':       mo2c_ws,
            'MoC wt%':          moc_wf,
            'MoC sigma':        moc_ws,
            'MoC variant':      moc_var,
            'Active phases':    ', '.join(result.get('active_phases', [])),
            'Rwp':              result['Rwp'],
            'GOF':              result['GOF'],
            'Chi2':             result['chi2'],
        }
        rows.append(row)

        print(f'\nResult for {pattern_name}:')
        print(f'  Mo: {mo_wf}% ± {mo_ws}%')
        print(f'  Mo2C: {mo2c_wf}% ± {mo2c_ws}%')
        print(f'  MoC ({moc_var}): {moc_wf}% ± {moc_ws}%')
        print(f'  Rwp={result["Rwp"]}%  GOF={result["GOF"]}  Chi2={result["chi2"]}')

    # Write CSV
    if rows:
        fieldnames = ['Index', 'Mo (Im3m) wt%', 'Mo sigma',
                      'Mo2C (ortho) wt%', 'Mo2C sigma',
                      'MoC wt%', 'MoC sigma', 'MoC variant',
                      'Active phases', 'Rwp', 'GOF', 'Chi2']
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n✅ Results saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
