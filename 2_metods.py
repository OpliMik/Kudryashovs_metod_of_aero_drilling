"""Working M1/M2 project calculation: dry air and cuttings, 2026-10-25.
Single Python file, standard library only. Constant prescribed temperature.
M1 uses seven attached-source functions verbatim. M2 uses Guo-Liu pressure
ODEs on the same four channels and shared resistance closure.
No liquid, thermal-balance, separate nozzle or sonic-flow subsystem.
N uses common source-compatible postprocessing, not independent Guo power.
Reference: Guo and Liu 2011, Chapter6, DOI10.1016/B978-0-12-381957-4.00006-1.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import math
import unittest


def require_finite(*values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError('All numeric inputs must be finite')

@dataclass(frozen=True)
class Gas:
    R: float = 287.0                    # J/(kg K)
    k: float = 1.4
    g: float = 9.81
    reference_pressure_pa: float = 14.7 * 6894.757293168
    reference_temperature_k: float = 520.0 * 5.0 / 9.0

    def __post_init__(self):
        require_finite(*self.__dict__.values())
        if min(self.R, self.g, self.reference_pressure_pa,
               self.reference_temperature_k) <= 0 or self.k <= 1:
            raise ValueError('Invalid gas or reference conditions')

    def mass_flow(self, reference_m3_min):
        require_finite(reference_m3_min)
        if reference_m3_min <= 0:
            raise ValueError('Positive reference gas flow required')
        return (reference_m3_min / 60 * self.reference_pressure_pa
                / (self.R * self.reference_temperature_k))

@dataclass(frozen=True)
class Section:
    """Top-to-bottom geometry. Angle is FROM VERTICAL, unlike original file.

    Constant area, hydraulic diameter, friction and mass loading per section.
    Temperature varies linearly along measured length. Curves must be split.
    Friction must be supplied explicitly, e.g. via nikuradse().
    """
    name: str
    length_m: float
    area_m2: float
    hydraulic_diameter_m: float
    friction_darcy: float
    top_temperature_k: float
    bottom_temperature_k: float
    inclination_from_vertical_deg: float = 0.0

    def __post_init__(self):
        require_finite(self.length_m, self.area_m2, self.hydraulic_diameter_m,
                       self.friction_darcy, self.top_temperature_k,
                       self.bottom_temperature_k, self.inclination_from_vertical_deg)
        vals = (self.length_m, self.area_m2, self.hydraulic_diameter_m,
                self.top_temperature_k, self.bottom_temperature_k)
        if min(vals) <= 0 or self.friction_darcy < 0:
            raise ValueError('Invalid section geometry, temperature or friction')
        if not 0 <= self.inclination_from_vertical_deg <= 90:
            raise ValueError('This revision supports 0..90 degrees from vertical')

def annulus(outer_diameter_m, inner_diameter_m):
    require_finite(outer_diameter_m, inner_diameter_m)
    if not 0 < inner_diameter_m < outer_diameter_m:
        raise ValueError('Require 0 < inner diameter < outer diameter')
    return (math.pi / 4 * (outer_diameter_m**2 - inner_diameter_m**2),
            outer_diameter_m - inner_diameter_m)

def nikuradse(roughness_m, hydraulic_diameter_m):
    """Eq. 6.19, fully rough turbulent Darcy friction, decimal logarithm.

    No automatic laminar/transitional replacement: that changes the closure.
    """
    require_finite(roughness_m, hydraulic_diameter_m)
    if not 0 < roughness_m < hydraulic_diameter_m / 2:
        raise ValueError('Require 0 < roughness < Dh/2')
    return (1.74 - 2 * math.log10(2 * roughness_m / hydraulic_diameter_m))**-2

def pressure_transform(section, gas, mass_gas_kg_s,
                       carried_mass_kg_s=0.0, flow='up'):
    """SI form of Eqs. 6.2, 6.4, 6.5, 6.25; solve Y=p**2 exactly.

    dY/ds = 2*M*g*cos(I)*Y/(R*T) + sign*f*M*mg**2*R*T/(Dh*A**2)
    M = 1 + carried_mass/mg; sign=+1 for upward flow, -1 downward.
    Coordinates s increase DOWN the well for BOTH flow directions.
    Return F,J such that p_bottom**2 = F*p_top**2 + J.
    This solves the source differential equation, including G=0 and horizontal
    limits, instead of changing the model at its analytic singularities.
    """
    require_finite(mass_gas_kg_s, carried_mass_kg_s)
    if mass_gas_kg_s <= 0 or carried_mass_kg_s < 0 or flow not in ('up', 'down'):
        raise ValueError('Invalid mass flow or direction')
    M = 1 + carried_mass_kg_s / mass_gas_kg_s
    cosine = math.cos(math.radians(section.inclination_from_vertical_deg))
    if abs(cosine) < 1e-14:
        cosine = 0.0
    a = 2 * M * gas.g * cosine / gas.R
    b = ((1 if flow == 'up' else -1) * section.friction_darcy * M
         * mass_gas_kg_s**2 * gas.R
         / (section.hydraulic_diameter_m * section.area_m2**2))
    L, T0 = section.length_m, section.top_temperature_k
    q = (section.bottom_temperature_k - T0) / L
    if q == 0:
        x = a * L / T0
        F = math.exp(x)
        J = b * T0 * L if a == 0 else b * T0**2 * math.expm1(x) / a
    else:
        log_ratio = math.log1p(q * L / T0)
        x = a * log_ratio / q
        F = math.exp(x)
        denominator = 2 * q - a
        z = denominator * log_ratio / q
        integral = (log_ratio / q if abs(z) < 1e-12
                    else math.expm1(z) / denominator)
        J = b * T0**2 * F * integral
    return F, J

def section_pressure(known_pressure_pa, section, gas, mass_gas_kg_s,
                     carried_mass_kg_s=0.0, flow='up', known_end='top'):
    require_finite(known_pressure_pa)
    if known_pressure_pa <= 0 or known_end not in ('top', 'bottom'):
        raise ValueError('Positive absolute pressure and valid known end required')
    F, J = pressure_transform(section, gas, mass_gas_kg_s, carried_mass_kg_s, flow)
    y = (F * known_pressure_pa**2 + J if known_end == 'top'
         else (known_pressure_pa**2 - J) / F)
    if y <= 0 or not math.isfinite(y):
        raise ValueError('No positive pressure for these boundary conditions')
    return math.sqrt(y)

def state(p_pa, T_k, area_m2, mg, gas):
    require_finite(p_pa, T_k, area_m2, mg)
    if min(p_pa, T_k, area_m2, mg) <= 0:
        raise ValueError('Positive state inputs required')
    rho = p_pa / (gas.R * T_k)
    v = mg / (rho * area_m2)
    # Eqs. 6.40-6.45: reference AIR at 50 ft/s, not gas-specific threshold.
    rho_air_reference = (0.0765 * 0.45359237 / 0.3048**3)
    return {'pressure_pa': p_pa, 'temperature_k': T_k, 'density_kg_m3': rho,
            'gas_velocity_m_s': v,
            'kinetic_energy_index': rho*v*v/(rho_air_reference*15.24**2)}

def default_data():
    return {
        "brhl_diam_D": 0.093,
        "dpth_h": 300,
        "drilling_angle_alfa": 90,
        "out_diam_dril_pipe_d": 0.05,
        "in_diam_dril_pipe_d1": 0.039,
        "cnctn_type_j": 1,
        "diam_chnl_join_d3": 0.028,
        "pipe_lngth_lt": 4.5,
        "out_diam_core_pipe_dh": 0.089,
        "in_diam_core_pipe_db": 0.079,
        "core_pipe_lngth_lk": 5,
        "core_diam_dk": 0.073,
        "core_out_bk": 80,
        "rock_density_ro": 2300,
        "mech_spd_vm": 6,
        "tool_type_j1": 2,
        "gast_koef_kr": 1.2,
        "coef_spd_unvnss_k": 1.2,
        "coef_red_lift_frc_k1": 1.1,
        "well_temp_t": 290,
        "rqrd_up_spd_v1": 12,
        "comprsr_effency_ni": 0.3,
        "ATM_PRSSR_P0": 100000,
        "GAS_CONST_R": 287,
        "ACCLRTN_GRAV_g": 9.81,
        "AIR_ADBTC_INDX_k": 1.4
    }

def calc_koef_a0(j):
    if j == 1:
        return 2
    elif j == 2:
        return 1.5
    elif j == 3:
        return 0
    else:
        print("Ошибка: некорректное значение j")

def calc_pressure_up(Ph, lambd, kr, nu, a, l, b):
    return (math.sqrt(Ph ** 2 + lambd * (1 + kr * nu) * a * l) + Ph * (1 + nu) * b * l) / (
                1 - (1 - nu) * b * l) * 10 ** (-6)

def calc_pressure_down(Ph, lambd, a, l, le, b):
    return (math.sqrt(Ph ** 2 + lambd * a * (l + le) / (1 + b * l)) - ((Ph * b * l) / (1 + b * l))) * 10 ** (-6)

def calc_an(G, R, T, Den, Fn):
    return (G ** 2 * R * T / (Den * Fn ** 2))

def calc_lambdn(Den):
    return (0.009407 / Den ** (1 / 3))

def hydraulic_model(data, G):
    brhl_diam_D = data["brhl_diam_D"]
    dpth_h = data["dpth_h"]
    drilling_angle_alfa = data["drilling_angle_alfa"]
    out_diam_dril_pipe_d = data["out_diam_dril_pipe_d"]
    in_diam_dril_pipe_d1 = data["in_diam_dril_pipe_d1"]
    cnctn_type_j = data["cnctn_type_j"]
    diam_chnl_join_d3 = data["diam_chnl_join_d3"]
    pipe_lngth_lt = data["pipe_lngth_lt"]
    out_diam_core_pipe_dh = data["out_diam_core_pipe_dh"]
    in_diam_core_pipe_db = data["in_diam_core_pipe_db"]
    core_pipe_lngth_lk = data["core_pipe_lngth_lk"]
    core_diam_dk = data["core_diam_dk"]
    core_out_bk = data["core_out_bk"]
    rock_density_ro = data["rock_density_ro"]
    mech_spd_vm = data["mech_spd_vm"]
    tool_type_j1 = data["tool_type_j1"]
    gast_koef_kr = data["gast_koef_kr"]
    coef_spd_unvnss_k = data["coef_spd_unvnss_k"]
    coef_red_lift_frc_k1 = data["coef_red_lift_frc_k1"]
    well_temp_t = data["well_temp_t"]
    rqrd_up_spd_v1 = data["rqrd_up_spd_v1"]
    comprsr_effency_ni = data["comprsr_effency_ni"]
    ATM_PRSSR_P0 = data["ATM_PRSSR_P0"]
    GAS_CONST_R = data["GAS_CONST_R"]
    ACCLRTN_GRAV_g = data["ACCLRTN_GRAV_g"]
    AIR_ADBTC_INDX_k = data["AIR_ADBTC_INDX_k"]
    drill_pipe_lngth_hbt = dpth_h - core_pipe_lngth_lk
    a0 = calc_koef_a0(cnctn_type_j)
    f1 = math.pi * (brhl_diam_D ** 2 - out_diam_dril_pipe_d ** 2) / 4
    f2 = math.pi * in_diam_dril_pipe_d1 ** 2 / 4
    f3 = math.pi * (brhl_diam_D ** 2 - out_diam_core_pipe_dh ** 2) / 4
    f4 = math.pi * (in_diam_core_pipe_db ** 2 - core_diam_dk ** 2) / 4
    De1 = brhl_diam_D - out_diam_dril_pipe_d
    De2 = in_diam_dril_pipe_d1
    De3 = brhl_diam_D - out_diam_core_pipe_dh
    De4 = in_diam_core_pipe_db - core_diam_dk
    air_density_ron = ATM_PRSSR_P0 / (GAS_CONST_R * 273)
    air_cons_q = G
    mass_air_cons_g = air_cons_q * air_density_ron / 60
    mass_rock_face_gz = math.pi / 4 * (
                brhl_diam_D ** 2 - core_out_bk * core_diam_dk ** 2 / 100) * rock_density_ro * mech_spd_vm / 3600
    sludge_conc_nu = mass_rock_face_gz / mass_air_cons_g
    c = GAS_CONST_R * well_temp_t
    b = ACCLRTN_GRAV_g * math.sin(drilling_angle_alfa * math.pi / 180) / (2 * c)
    e = mass_air_cons_g ** 2 * c
    a1 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De1, f1)
    a2 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De2, f2)
    a3 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De3, f3)
    a4 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De4, f4)
    lambd1 = calc_lambdn(De1)
    lambd2 = calc_lambdn(De2)
    lambd3 = calc_lambdn(De3)
    lambd4 = calc_lambdn(De4)
    nc = math.ceil(drill_pipe_lngth_hbt / pipe_lngth_lt)
    csi = a0 * ((in_diam_dril_pipe_d1 / diam_chnl_join_d3) ** 2 - 1) ** 2
    le = in_diam_dril_pipe_d1 * nc * csi / lambd2
    P1 = calc_pressure_up(ATM_PRSSR_P0, lambd1, gast_koef_kr, sludge_conc_nu, a1, drill_pipe_lngth_hbt, b)
    P2 = calc_pressure_up(P1 * 10 ** 6, lambd3, gast_koef_kr, sludge_conc_nu, a3, core_pipe_lngth_lk, b)
    P3 = calc_pressure_down(P2 * 10 ** 6, lambd4, a4, core_pipe_lngth_lk, 0, b)
    P4 = calc_pressure_down(P3 * 10 ** 6, lambd2, a2, drill_pipe_lngth_hbt, le, b)
    Pres = P4 + 0.05
    N = AIR_ADBTC_INDX_k / (AIR_ADBTC_INDX_k - 1) * ATM_PRSSR_P0 * air_cons_q / 60 * (
                (Pres * 10 ** 6 / ATM_PRSSR_P0) ** (
                    (coef_spd_unvnss_k - 1) / coef_spd_unvnss_k) - 1) / comprsr_effency_ni
    vkol = air_cons_q / (60 * f1)
    vbt = air_cons_q / (60 * f2)
    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "Pres": Pres,
        "N": N,
        "vkol": vkol,
        "vbt": vbt
    }

def project_common(data, q_normal_m3_min=None):
    validate_project_scope_inputs(data)
    d = data
    qn = q_normal_m3_min
    if qn is None:
        qn = 47.1*d['coef_spd_unvnss_k']*d['coef_red_lift_frc_k1']*(d['brhl_diam_D']**2-d['out_diam_dril_pipe_d']**2)*d['rqrd_up_spd_v1']
    gas = Gas(R=d['GAS_CONST_R'], k=d['AIR_ADBTC_INDX_k'], g=d['ACCLRTN_GRAV_g'], reference_pressure_pa=d['ATM_PRSSR_P0'], reference_temperature_k=273.0)
    mg = gas.mass_flow(qn)
    ms = math.pi/4*(d['brhl_diam_D']**2-d['core_out_bk']/100*d['core_diam_dk']**2)*d['rock_density_ro']*d['mech_spd_vm']/3600
    return gas, qn, mg, ms

def project_channels(data, friction_mode='shared_kudryashov', epsilon_m=None,
                      include_joints=True):
    """Four source channels in backward pressure-calculation order.

    Mode shared_kudryashov: diagnostic, NOT native Guo friction closure.
    Mode uniform_roughness: Eq.6.19 using HYPOTHETICAL equal wall roughness.
    All walls have the same epsilon within one sensitivity scenario, so the
    perimeter average in Eq.6.20 is identically epsilon. No casing is added.

    Joint sum K = nc*xi is from the supplied source in both modes. Its
    resistance is spread along the pipe: f_eff=f+Dh*sum(K)/L. This is an
    explicit project adaptation, not a new claim about Guo's Excel workbook.
    Physical gravitational length stays L, not L+le.
    """
    d = data
    length_upper = d['dpth_h']-d['core_pipe_lngth_lk']
    if length_upper <= 0:
        raise ValueError('Depth must exceed core barrel length')
    a_ann, dh_ann = annulus(d['brhl_diam_D'], d['out_diam_dril_pipe_d'])
    a_core_ann, dh_core_ann = annulus(d['brhl_diam_D'], d['out_diam_core_pipe_dh'])
    a_core_in, dh_core_in = annulus(d['in_diam_core_pipe_db'], d['core_diam_dk'])
    raw = [
        ('annulus_around_drill_pipe', length_upper, a_ann, dh_ann, 'up', 0, length_upper),
        ('annulus_around_core_barrel', d['core_pipe_lngth_lk'], a_core_ann, dh_core_ann, 'up', length_upper, d['dpth_h']),
        ('inside_core_barrel_around_core', d['core_pipe_lngth_lk'], a_core_in, dh_core_in, 'down', length_upper, d['dpth_h']),
        ('inside_drill_pipe', length_upper, math.pi/4*d['in_diam_dril_pipe_d1']**2, d['in_diam_dril_pipe_d1'], 'down', 0, length_upper),
    ]
    count = math.ceil(length_upper/d['pipe_lngth_lt'])
    xi = calc_koef_a0(d['cnctn_type_j'])*((d['in_diam_dril_pipe_d1']/d['diam_chnl_join_d3'])**2-1)**2
    joint_sum_k = count*xi if include_joints else 0.0
    channels = []
    for name, L, area, Dh, flow, top_depth, bottom_depth in raw:
        if friction_mode == 'shared_kudryashov':
            f_wall = calc_lambdn(Dh)
        elif friction_mode == 'uniform_roughness':
            if epsilon_m is None:
                raise ValueError('Explicit wall roughness is required')
            f_wall = nikuradse(epsilon_m, Dh)
        else:
            raise ValueError('Unknown friction mode')
        f_joint = Dh*joint_sum_k/L if name == 'inside_drill_pipe' else 0.0
        s = Section(name, L, area, Dh, f_wall+f_joint, d['well_temp_t'], d['well_temp_t'], 90-d['drilling_angle_alfa'])
        channels.append({'section': s, 'flow': flow, 'wall_friction': f_wall,
                         'joint_friction_equivalent': f_joint,
                         'top_depth_m': top_depth, 'bottom_depth_m': bottom_depth})
    return channels, {'connection_count': count, 'xi_each': xi,
                      'joint_sum_K': joint_sum_k,
                      'source_equivalent_length_m': d['in_diam_dril_pipe_d1']*count*xi/calc_lambdn(d['in_diam_dril_pipe_d1'])}

def calculate_project_guo(data, q_normal_m3_min=None,
                          friction_mode='shared_kudryashov', epsilon_m=None,
                          include_joints=True):
    """Guo differential pressure equation on the source's FOUR-CHANNEL layout.

    Dry, isothermal; no separate bit loss, junction acceleration jump or
    surface line. This is NOT a reproduction of Example 6.4 or its bit.
    Minimum normal flow is NOT imposed by this function.
    """
    gas, qn, mg, ms = project_common(data, q_normal_m3_min)
    channels, joints = project_channels(data, friction_mode, epsilon_m, include_joints)
    p = data['ATM_PRSSR_P0']
    pressures, profiles, section_data = [], [], []
    for item in channels:
        s, flow = item['section'], item['flow']
        carried = ms if flow == 'up' else 0.0
        known_end = 'top' if flow == 'up' else 'bottom'
        unknown_end = 'bottom' if flow == 'up' else 'top'
        p_start = p
        p = section_pressure(p, s, gas, mg, carried, flow, known_end)
        p_top, p_bottom = ((p_start, p) if flow == 'up' else (p, p_start))
        pressures.append(p)
        section_data.append({'section': s.name, 'length_m': s.length_m,
                             'area_m2': s.area_m2, 'hydraulic_diameter_m': s.hydraulic_diameter_m,
                             'wall_friction': item['wall_friction'],
                             'joint_friction_equivalent': item['joint_friction_equivalent'],
                             'top_pressure_pa': p_top, 'bottom_pressure_pa': p_bottom})
        # Include both sides of every geometry transition as separate states.
        # For vertical, isothermal upflow: p strictly increases down-section,
        # so KEI proportional to 1/p reaches its section minimum at the bottom.
        for j in range(101):
            fraction = j/100
            if j == 0:
                local_pressure = p_top
            else:
                local = replace(s, length_m=s.length_m*fraction)
                local_pressure = section_pressure(p_top, local, gas, mg, carried, flow)
            depth = item['top_depth_m']+(item['bottom_depth_m']-item['top_depth_m'])*fraction
            profiles.append({'section': s.name, 'flow': flow, 'depth_m': depth,
                             **state(local_pressure, data['well_temp_t'], s.area_m2, mg, gas)})
    annular_points = [p for p in profiles if p['flow']=='up']
    minimum = min(annular_points, key=lambda p:p['kinetic_energy_index'])
    result = {f'P{i+1}_MPa': p/1e6 for i, p in enumerate(pressures)}
    result.update({'friction_mode': friction_mode, 'epsilon_uniform_mm': None if epsilon_m is None else epsilon_m*1000,
                   'q_normal_m3_min_at_273K': qn, 'mass_gas_kg_s': mg,
                   'mass_solids_kg_s': ms, 'joint_data': joints,
                   'bit_pressure_drop_pa': 0.0,
                   'bit_loss_status': 'OMITTED, not physically measured zero',
                   'legacy_margin_MPa': .05,
                   'P4_plus_legacy_margin_MPa': pressures[-1]/1e6+.05,
                   'minimum_annular_KEI': minimum['kinetic_energy_index'],
                   'minimum_KEI_section': minimum['section'],
                   'minimum_KEI_depth_m': minimum['depth_m'],
                   'minimum_KEI_gas_velocity_m_s': minimum['gas_velocity_m_s'],
                   'power_comparison_status': 'Not calculated: stage count and matched thermodynamic model require separate audit',
                   'sections': section_data, 'profiles': profiles})
    return result

def normal_flow_for_kei(data, target_kei, friction_mode='shared_kudryashov', epsilon_m=None):
    """Optional KEI diagnostic for this dry/isothermal four-channel adaptation.
    Endpoint minimum is exact for the annular section equations used here.
    Does not resolve bed formation, individual slip or nozzle performance.
    """
    if target_kei <= 0:
        raise ValueError('Positive KEI target required')
    def kei(q):
        gas, qn, mg, ms = project_common(data, q)
        channels, _ = project_channels(data, friction_mode, epsilon_m)
        p = data['ATM_PRSSR_P0']
        minimum = float('inf')
        for item in channels[:2]:
            s = item['section']
            minimum = min(minimum, state(p, data['well_temp_t'], s.area_m2, mg, gas)['kinetic_energy_index'])
            p = section_pressure(p, s, gas, mg, ms, 'up')
            minimum = min(minimum, state(p, data['well_temp_t'], s.area_m2, mg, gas)['kinetic_energy_index'])
        return minimum
    lo, hi = .01, project_common(data)[1]
    while kei(hi) < target_kei:
        hi *= 2
        if hi > 1000:
            raise ValueError('KEI root not bracketed below 1000 m3/min')
    if kei(lo) >= target_kei:
        raise ValueError('Lower bound already meets target')
    for _ in range(70):
        mid = (lo+hi)/2
        if kei(mid) < target_kei:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def validate_project_scope_inputs(data):
    """Dry project scope, exact source input names and units. No extra phases.
    Angle remains FROM HORIZONTAL at the public project interface.
    """
    required=default_data()
    missing=sorted(set(required)-set(data))
    if missing:raise ValueError('Missing project inputs: '+', '.join(missing))
    require_finite(*(data[key] for key in required))
    for key in ('mist_mass_kg_s','formation_liquid_mass_kg_s','mist_bbl_hr','formation_liquid_bbl_hr'):
        if key in data and data[key]!=0:
            raise ValueError('Working M2 scope is dry air and cuttings; liquid input is unsupported')
    positive=('brhl_diam_D','dpth_h','out_diam_dril_pipe_d','in_diam_dril_pipe_d1',
              'diam_chnl_join_d3','pipe_lngth_lt','out_diam_core_pipe_dh','in_diam_core_pipe_db',
              'core_pipe_lngth_lk','core_diam_dk','rock_density_ro','well_temp_t',
              'gast_koef_kr','coef_spd_unvnss_k','coef_red_lift_frc_k1','rqrd_up_spd_v1',
              'ATM_PRSSR_P0','GAS_CONST_R','ACCLRTN_GRAV_g')
    if min(data[key] for key in positive)<=0:raise ValueError('Positive project dimensions/properties required')
    if not (0<=data['drilling_angle_alfa']<=90 and 0<=data['core_out_bk']<=100
            and data['mech_spd_vm']>=0 and 0<data['comprsr_effency_ni']<=1
            and data['AIR_ADBTC_INDX_k']>1 and data['cnctn_type_j'] in (1,2,3)):
        raise ValueError('Invalid project inclination, recovery, rates or coefficients')
    if not (data['dpth_h']>data['core_pipe_lngth_lk']
            and data['brhl_diam_D']>max(data['out_diam_dril_pipe_d'],data['out_diam_core_pipe_dh'])
            and data['out_diam_dril_pipe_d']>data['in_diam_dril_pipe_d1']>=data['diam_chnl_join_d3']
            and data['out_diam_core_pipe_dh']>data['in_diam_core_pipe_db']>data['core_diam_dk']):
        raise ValueError('Project channel dimensions do not fit')

def shared_project_outputs(data,q_normal_m3_min,pressures_MPa):
    """Same output schema and common postprocessing as the attached source.

    N is source-compatible postprocessing, not independent Guo thermodynamics.
    The source's mixed exponents are retained explicitly for controlled
    pressure-equation comparison. Research compressor alternatives are separate.
    vkol/vbt use REFERENCE volume and are not local gas velocities.
    """
    P1,P2,P3,P4=pressures_MPa;Pres=P4+.05
    k=data['AIR_ADBTC_INDX_k'];exponent_k=data['coef_spd_unvnss_k'];p0=data['ATM_PRSSR_P0']
    if exponent_k<=0:raise ValueError('Positive legacy power exponent parameter required')
    N=k/(k-1)*p0*q_normal_m3_min/60*((Pres*1e6/p0)**((exponent_k-1)/exponent_k)-1)/data['comprsr_effency_ni']
    annular_area=math.pi/4*(data['brhl_diam_D']**2-data['out_diam_dril_pipe_d']**2)
    internal_area=math.pi/4*data['in_diam_dril_pipe_d1']**2
    return {'P1':P1,'P2':P2,'P3':P3,'P4':P4,'Pres':Pres,'N':N,
            'vkol':q_normal_m3_min/(60*annular_area),'vbt':q_normal_m3_min/(60*internal_area)}

def hydraulic_model_m2(data,G):
    """Working M2 interface matching hydraulic_model(data,G) and its 8 keys.

    G is reference m3/min at 273 K and data['ATM_PRSSR_P0'], NOT kg/s.
    Dry gas and cuttings; constant prescribed T; same four-channel geometry.
    Shared resistance closure isolates Guo vs Kudryashov pressure equations.
    No separate nozzle loss or new thermal/liquid inputs are required.
    N uses an explicitly shared legacy postprocessor, not native Guo power.
    """
    result=calculate_project_guo(data,G)
    return shared_project_outputs(data,G,[result[f'P{i}_MPa'] for i in range(1,5)])

class ProjectScopeVerification(unittest.TestCase):
    def test_matching_interface_and_attached_baseline(self):
        d=default_data();_,q,_,_=project_common(d)
        m1=hydraulic_model(d,q);m2=hydraulic_model_m2(d,q)
        self.assertEqual(set(m1),set(m2))
        self.assertAlmostEqual(m1['P4'],.677728215777318,places=12)
        self.assertAlmostEqual(m2['P4'],.666622917375,places=11)
        for key in ('vkol','vbt'):self.assertEqual(m1[key],m2[key])
        self.assertAlmostEqual(m2['Pres']-m2['P4'],.05,places=14)
        common=shared_project_outputs(d,q,[m1[f'P{i}'] for i in range(1,5)])
        self.assertEqual(common,m1)

    def test_dry_scope_needs_no_nozzle_or_thermal_extra_inputs(self):
        d=default_data();before=json.dumps(d,sort_keys=True)
        gas,q,mg,ms=project_common(d);r=calculate_project_guo(d,q)
        self.assertGreater(ms,0);self.assertGreater(mg,0)
        self.assertEqual(r['bit_pressure_drop_pa'],0)
        self.assertEqual(before,json.dumps(d,sort_keys=True))
        wet=dict(d,mist_bbl_hr=1)
        with self.assertRaises(ValueError):hydraulic_model_m2(wet,q)

    def test_geometry_and_nonfinite_inputs(self):
        for key,value in (('core_out_bk',120),('well_temp_t',math.nan),('in_diam_core_pipe_db',.072)):
            d=default_data();d[key]=value
            with self.assertRaises(ValueError):hydraulic_model_m2(d,4.)

    def test_mass_and_joint_resistance_preserved(self):
        d=default_data();gas,q,mg,ms=project_common(d);channels,joints=project_channels(d)
        self.assertAlmostEqual(mg,q/60*1e5/(287*273),places=14)
        self.assertAlmostEqual(ms,math.pi/4*(.093**2-.8*.073**2)*2300*6/3600,places=14)
        joint_section=channels[-1]
        self.assertAlmostEqual(joint_section['joint_friction_equivalent']*joint_section['section'].length_m/joint_section['section'].hydraulic_diameter_m,
                               joints['joint_sum_K'],places=12)
        with_joints=calculate_project_guo(d,q);without=calculate_project_guo(d,q,include_joints=False)
        self.assertGreater(with_joints['P4_MPa'],without['P4_MPa'])
        self.assertEqual(with_joints['P2_MPa'],without['P2_MPa'])

    def test_same_conditions_across_27_scenarios(self):
        d=default_data();q0=project_common(d)[1]
        for depth in (100.,200.,300.):
            for ROP in (3.,6.,9.):
                for q in (3.,q0,6.):
                    case=dict(d,dpth_h=depth,mech_spd_vm=ROP)
                    first=hydraulic_model(case,q);second=hydraulic_model_m2(case,q)
                    self.assertLess(abs(second['P4']/first['P4']-1),.025)
                    self.assertEqual(first['vkol'],second['vkol'])

class ProjectVerification(unittest.TestCase):
    def test_attached_baseline(self):
        d = default_data()
        q = project_common(d)[1]
        r = hydraulic_model(d,q)
        self.assertAlmostEqual(r['P4'], .677728215777318, places=12)

    def test_joint_resistance_preserved(self):
        channels,j = project_channels(default_data())
        s = channels[-1]['section']
        added = channels[-1]['joint_friction_equivalent']
        self.assertAlmostEqual(added*s.length_m/s.hydraulic_diameter_m, j['joint_sum_K'], places=12)
        with_j = calculate_project_guo(default_data())
        without = calculate_project_guo(default_data(),include_joints=False)
        self.assertGreater(with_j['P4_MPa'], without['P4_MPa'])
        self.assertEqual(with_j['P2_MPa'], without['P2_MPa'])

    def test_source_defaults_not_mutated(self):
        d = default_data()
        before = json.dumps(d,sort_keys=True)
        calculate_project_guo(d)
        self.assertEqual(before,json.dumps(d,sort_keys=True))

    def test_kei_root(self):
        d = default_data()
        q = normal_flow_for_kei(d,1.2)
        r = calculate_project_guo(d,q)
        self.assertAlmostEqual(r['minimum_annular_KEI'],1.2,places=12)

    def test_all_four_sections_against_direct_rk4(self):
        d = default_data()
        gas,_,mg,ms = project_common(d)
        for mode,eps in [('shared_kudryashov',None),('uniform_roughness',.00005)]:
            channels,_ = project_channels(d,mode,eps)
            result = calculate_project_guo(d,friction_mode=mode,epsilon_m=eps)
            for item, row in zip(channels,result['sections']):
                s,flow = item['section'],item['flow']
                loading = 1+(ms/mg if flow=='up' else 0)
                sign = 1 if flow=='up' else -1
                cosine = math.cos(math.radians(s.inclination_from_vertical_deg))
                def derivative(p):
                    rho = p/(gas.R*d['well_temp_t'])
                    v = mg/(rho*s.area_m2)
                    return loading*rho*(gas.g*cosine+sign*s.friction_darcy*v*v/(2*s.hydraulic_diameter_m))
                # Integrate p itself from its known boundary to avoid sharing the
                # affine p-squared implementation and to keep boundary known.
                p = row['top_pressure_pa'] if flow=='up' else row['bottom_pressure_pa']
                dx = s.length_m/10000*(1 if flow=='up' else -1)
                for _ in range(10000):
                    k1=derivative(p); k2=derivative(p+dx*k1/2)
                    k3=derivative(p+dx*k2/2); k4=derivative(p+dx*k3)
                    p += dx/6*(k1+2*k2+2*k3+k4)
                expected = row['bottom_pressure_pa'] if flow=='up' else row['top_pressure_pa']
                self.assertLess(abs(p/expected-1),1e-10)

def run_project_scope(output_directory):
    """Default research run for user's actual project, no new physical subsystems."""
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    data=default_data();gas,q,mg,ms=project_common(data)
    m1=hydraulic_model(data,q);m2=hydraulic_model_m2(data,q)
    details=calculate_project_guo(data,q)
    details.pop('power_comparison_status',None)
    rows=[]
    for depth in (100.,200.,300.):
        for ROP in (3.,6.,9.):
            for Q in (3.,q,6.):
                d=dict(data,dpth_h=depth,mech_spd_vm=ROP)
                first=hydraulic_model(d,Q);second=hydraulic_model_m2(d,Q)
                rows.append({'depth_m':depth,'penetration_m_h':ROP,'Q_reference_m3_min':Q,
                             'P4_Kudryashov_MPa_absolute':first['P4'],'P4_M2_MPa_absolute':second['P4'],
                             'difference_percent':100*(second['P4']/first['P4']-1)})
    payload={'scope':'Dry air plus cuttings, prescribed constant temperature, matching attached Kudryashov project',
             'source_defaults':data,'Q_reference_m3_min':q,'reference_temperature_K':273.,
             'reference_pressure_Pa':data['ATM_PRSSR_P0'],'mass_gas_kg_s':mg,'mass_solids_kg_s':ms,
             'M1_attached_Kudryashov':m1,'M2_Guo_pressure_equations_shared_resistance':m2,
             'pressure_comparison_scope':'Shared wall friction and joint K. This isolates pressure equations; native friction is a separate optional closure.',
             'power_comparison_scope':'N uses identical legacy postprocessing for both pressures. This is not independent Guo compressor power or validation of the source power formula.',
             'reference_velocities_note':'vkol/vbt use reference volume. Local velocities are in M2_details.profiles.',
             'required_new_liquid_inputs':[],'required_new_thermal_inputs':[],
             'thermal_or_wet_restriction_models_required_for_project_readiness':False,
             'bit_loss_note':'Omitted consistently with attached source; no assertion of measured physical zero',
             'M2_details':details,'paired_scenarios':rows,
             'difference_P4_percent':100*(m2['P4']/m1['P4']-1)}
    (out/'project_scope_results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'project_scope_scenarios.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'project_scope_verification.txt').open('w',encoding='utf-8') as f:
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ProjectVerification,ProjectScopeVerification))
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        if not result.wasSuccessful():raise RuntimeError('Project scope verification failed')
    return payload

# Console application. All calculations and tests are contained in this file.
FIELDS = [
    ('brhl_diam_D', 'Диаметр скважины', 'м'),
    ('dpth_h', 'Длина скважины по стволу', 'м'),
    ('drilling_angle_alfa', 'Угол от горизонтали (90 = вертикально)', 'град'),
    ('out_diam_dril_pipe_d', 'Наружный диаметр бурильной трубы', 'м'),
    ('in_diam_dril_pipe_d1', 'Внутренний диаметр бурильной трубы', 'м'),
    ('cnctn_type_j', 'Тип соединения (1: a0=2; 2: a0=1,5; 3: a0=0)', 'код'),
    ('diam_chnl_join_d3', 'Диаметр канала соединения', 'м'),
    ('pipe_lngth_lt', 'Длина бурильной трубы', 'м'),
    ('out_diam_core_pipe_dh', 'Наружный диаметр колонковой трубы', 'м'),
    ('in_diam_core_pipe_db', 'Внутренний диаметр колонковой трубы', 'м'),
    ('core_pipe_lngth_lk', 'Длина колонковой трубы', 'м'),
    ('core_diam_dk', 'Диаметр керна', 'м'),
    ('core_out_bk', 'Выход керна', '%'),
    ('rock_density_ro', 'Плотность породы', 'кг/м³'),
    ('mech_spd_vm', 'Механическая скорость бурения', 'м/ч'),
    ('tool_type_j1', 'Код инструмента (справочно, в формулах не используется)', 'код'),
    ('gast_koef_kr', 'Коэффициент kr в методе Кудряшова', ''),
    ('coef_spd_unvnss_k', 'Коэффициент неравномерности скорости k', ''),
    ('coef_red_lift_frc_k1', 'Коэффициент снижения подъёмной силы k1', ''),
    ('well_temp_t', 'Постоянная температура (290 K = 16,85 °C)', 'K'),
    ('rqrd_up_spd_v1', 'Требуемая восходящая скорость для выбора расхода', 'м/с'),
    ('comprsr_effency_ni', 'КПД компрессора (0 < КПД <= 1)', 'доля'),
    ('ATM_PRSSR_P0', 'Атмосферное абсолютное давление', 'Па'),
    ('GAS_CONST_R', 'Газовая постоянная воздуха', 'Дж/(кг·K)'),
    ('ACCLRTN_GRAV_g', 'Ускорение свободного падения', 'м/с²'),
    ('AIR_ADBTC_INDX_k', 'Показатель адиабаты', ''),
]


def number(text):
    value = float(text.strip().replace(',', '.'))
    require_finite(value)
    return value


def validate_session(data, flow):
    validate_project_scope_inputs(data)
    if data['tool_type_j1'] != int(data['tool_type_j1']):
        raise ValueError('Код инструмента должен быть целым числом.')
    project_common(data, flow)


def show_inputs(data, flow):
    print('\nИСХОДНЫЕ ДАННЫЕ (диаметры вводятся в метрах)')
    for i, (key, label, unit) in enumerate(FIELDS, 1):
        print(f'{i:2}. {label}: {data[key]:g} {unit}')
    q = project_common(data, flow)[1]
    mode = 'по исходной формуле Кудряшова' if flow is None else 'задан вручную'
    print(f'\nОбщий расход Q = {q:.9g} м³/мин, {mode}.')
    print(f"Условия приведения: 273 K, {data['ATM_PRSSR_P0']:g} Па.")


def edit_inputs(data):
    candidate = dict(data)
    print('\nМожно изменить несколько параметров, затем проверить и применить вместе.')
    print('Номер параметра: новое значение. Пустой ввод сохраняет значение.')
    while True:
        for i, (key, label, unit) in enumerate(FIELDS, 1):
            print(f'{i:2}. {label}: {candidate[key]:g} {unit}')
        choice = input('Номер; s = проверить и сохранить; x = отменить: ').strip().lower()
        if choice == 'x':
            return data
        if choice == 's':
            try:
                validate_session(candidate, None)
            except (ValueError, OverflowError) as exc:
                print(f'Данные не сохранены: {exc}\nПроверьте размеры, диапазоны и единицы.')
                continue
            return candidate
        try:
            index = int(choice)
            if not 1 <= index <= len(FIELDS):
                raise ValueError('Нет такого параметра.')
            key, label, unit = FIELDS[index-1]
            raw = input(f'{label} [{candidate[key]:g}] {unit}: ').strip()
            if raw:
                candidate[key] = number(raw)
        except ValueError as exc:
            print(f'Ошибка ввода: {exc}')


def calculate_pair(data, flow=None):
    validate_session(data, flow)
    q = project_common(data, flow)[1]
    first, second = hydraulic_model(data, q), hydraulic_model_m2(data, q)
    for result in (first, second):
        require_finite(*result.values())
        if min(result[key] for key in ('P1', 'P2', 'P3', 'P4', 'Pres')) <= 0:
            raise ValueError('Получено неположительное давление; расчёт неприменим.')
    return q, first, second


def show_results(data, flow, method='both'):
    q, first, second = calculate_pair(data, flow)
    print(f'\nОБЩИЙ РАСХОД: {q:.9g} м³/мин при 273 K и {data["ATM_PRSSR_P0"]:g} Па')
    print('Давления абсолютные. N показана в кВт.')
    labels = {'P1':'P1, затрубье над колонковой трубой, МПа',
              'P2':'P2, забой снаружи, МПа', 'P3':'P3, верх внутри колонковой трубы, МПа',
              'P4':'P4, вход в бурильную колонну, МПа',
              'Pres':'Pres = P4 + 0,05, МПа', 'N':'N, общий расчёт мощности, кВт',
              'vkol':'vkol, по приведённому расходу, м/с',
              'vbt':'vbt, по приведённому расходу, м/с'}
    print(f'{"Показатель":46} {"Кудряшов":>13} {"Гуо (адапт.)":>13} {"Разница, %":>12}')
    for key, label in labels.items():
        a, b = first[key], second[key]
        scale = 1000 if key == 'N' else 1
        delta = f'{100*(b-a)/a:.4f}' if a != 0 else 'не определена'
        left = f'{a/scale:.6f}' if method != 'm2' else ''
        right = f'{b/scale:.6f}' if method != 'm1' else ''
        print(f'{label:46} {left:>13} {right:>13} {delta if method == "both" else "":>12}')
    print('Разница = (Гуо - Кудряшов) / Кудряшов × 100%.')
    print('Гуо: уравнения давления с общими сопротивлениями; сухой воздух и шлам.')
    print('N: общая исходная формула, не независимая модель мощности Гуо.')
    print('vkol/vbt не являются местными скоростями сжатого воздуха.')
    if first['N'] <= 0 or second['N'] <= 0:
        print('ВНИМАНИЕ: неположительная мощность. Проверьте k и область применимости исходной формулы.')


def export_results(data, flow, destination):
    q, first, second = calculate_pair(data, flow)
    payload = {'schema_version': 1, 'inputs': data, 'flow_override_m3_min': flow,
               'Q_reference_m3_min': q, 'reference_temperature_K': 273,
               'reference_pressure_Pa': data['ATM_PRSSR_P0'],
               'units': {'pressure': 'MPa absolute', 'N': 'W', 'velocity': 'reference m/s'},
               'M1_Kudryashov': first, 'M2_Guo_adapted': second,
               'notes': ['Dry air and cuttings; constant temperature; shared friction and joint resistance.',
                         'Power uses the shared original postprocessor, not independent Guo thermodynamics.']}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def input_path(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip().strip('"')
    return Path(raw).expanduser() if raw else Path(__file__).resolve().parent / default


def main():
    data, flow = default_data(), None
    print('КУДРЯШОВ И ГУО: расчёт сухой продувки')
    print('Python 3.10+, дополнительные библиотеки не нужны. Дроби: точка или запятая.')
    while True:
        print('\n1. Показать исходные данные\n2. Изменить исходные данные\n'
              '3. Выбрать общий расход воздуха\n4. Расчёт Кудряшова\n'
              '5. Расчёт Гуо\n6. Рассчитать и сравнить оба метода\n'
              '7. Сохранить исходные данные\n8. Загрузить исходные данные\n'
              '9. Сохранить результаты обоих методов (JSON)\n'
              '10. Вернуть исходные значения\n11. Выполнить встроенные проверки\n0. Выход')
        try:
            choice = input('Выберите пункт: ').strip()
            if choice == '0':
                return
            if choice == '1':
                show_inputs(data, flow)
            elif choice == '2':
                data = edit_inputs(data)
            elif choice == '3':
                raw = input('Расход, м³/мин; a = автоматический; Enter = без изменений: ').strip()
                if raw:
                    candidate = None if raw.lower() == 'a' else number(raw)
                    validate_session(data, candidate)
                    flow = candidate
                    show_inputs(data, flow)
            elif choice in ('4', '5', '6'):
                show_results(data, flow, {'4':'m1', '5':'m2', '6':'both'}[choice])
            elif choice == '7':
                path = input_path('Файл исходных данных', 'inputs.json')
                path.write_text(json.dumps({'schema_version':1, 'inputs':data,
                                           'flow_override_m3_min':flow}, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'Сохранено: {path.resolve()}')
            elif choice == '8':
                path = input_path('Файл исходных данных', 'inputs.json')
                payload = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(payload, dict) or payload.get('schema_version') != 1:
                    raise ValueError('Ожидается файл данных этой программы, версия 1.')
                candidate = payload['inputs']
                if not isinstance(candidate, dict) or set(candidate) != set(default_data()):
                    raise ValueError('Набор исходных параметров не соответствует программе.')
                if any(type(v) not in (int, float) for v in candidate.values()):
                    raise ValueError('Все исходные параметры должны быть числами.')
                qnew = payload.get('flow_override_m3_min')
                if qnew is not None and type(qnew) not in (int, float):
                    raise ValueError('Расход должен быть числом или null.')
                validate_session(candidate, qnew)
                data, flow = dict(candidate), qnew
                print('Исходные данные загружены и проверены.')
            elif choice == '9':
                path = input_path('Файл результатов', 'results.json')
                export_results(data, flow, path)
                print(f'Сохранено: {path.resolve()}')
            elif choice == '10':
                data, flow = default_data(), None
                print('Восстановлены исходные данные приложенного файла Кудряшова.')
            elif choice == '11':
                suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                                           for c in (ProjectVerification, ProjectScopeVerification))
                result = unittest.TextTestRunner(verbosity=2).run(suite)
                print('Проверки пройдены.' if result.wasSuccessful() else 'Есть ошибки проверок.')
            else:
                print('Нет такого пункта меню.')
        except (EOFError, KeyboardInterrupt):
            print('\nРабота завершена.')
            return
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError) as exc:
            print(f'Операция не выполнена: {exc}')
            print('Проверьте ввод, геометрию каналов, единицы или путь к файлу.')


if __name__ == '__main__':
    main()
