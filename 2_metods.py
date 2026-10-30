"""Dry air flushing: M1/M2/M3 with numerical quality controls, 2026-10-29.
One file, stdlib. Stationary M3 section and conditional full-loop solver.
Scientific validation remains explicitly separate from numerical verification.
M1: AIR, Gorshkov/Klimov/Kozlov/Yakovlev, Leningrad Mining Institute 1990,
pp.27-36, BASIC lines400-850. Mikhailova1985 VI.26 sign is a diagnostic only.
M2: Guo-Liu2011 Ch6, DOI10.1016/B978-0-12-381957-4.00006-1, pressure ODE.
Shared friction is a controlled project adaptation, not GasDrill-09.xls.
Book reproduction and common-condition comparison have distinct labels.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import math
import unittest
import html
from datetime import datetime, timezone, timedelta


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
    return (math.pi / 4 * ((outer_diameter_m-inner_diameter_m)*(outer_diameter_m+inner_diameter_m)),
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
        "AIR_ADBTC_INDX_k": 1.4,
        "flow_reference_temperature_k": 273.,
        "compressor_intake_temperature_k": 293.15,
        "receiver_margin_pa": 50000.,
        "air_power_N0": 0.,
        "source_literal_mode": 0,
        "up_formula_variant": 1,
        "geometry_mode": 1,
        "connection_count_override": 0.,
        "guo_friction_mode": 0,
        "pipe_roughness_m": 0.,
        "borehole_roughness_m": 0.,
        "core_roughness_m": 0.,
        "air_substeps": 1
    }

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
    ('tool_type_j1', 'Инструмент: 1 алмазный; 2 твердосплавный; 3 долото (kr вводится отдельно)', 'код'),
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

FIELDS += [
 ('pipe_roughness_m', 'Шероховатость стальных стенок; 0 = неизвестна', 'м'),
 ('borehole_roughness_m', 'Шероховатость стенки скважины; 0 = неизвестна', 'м'),
 ('core_roughness_m', 'Шероховатость поверхности керна; 0 = неизвестна', 'м'),
 ('guo_friction_mode', 'Guo: 0 общее трение; 1 Nikuradse; 2 Colebrook (модификация), сначала задать шероховатости', 'код'),
 ('flow_reference_temperature_k', 'Температура приведения объёмного расхода', 'K'),
 ('compressor_intake_temperature_k', 'Температура воздуха на входе компрессора', 'K'),
 ('receiver_margin_pa', 'Добавка давления наземной обвязки (предположение)', 'Па'),
 ('air_power_N0', 'Табличный N0 AIR; 0 = неизвестен, мощность AIR не вычислять', ''),
 ('source_literal_mode', '1 = буквальный AIR; 0 = общие условия сравнения', 'код'),
 ('up_formula_variant', 'Вариант M1: 1 AIR; 2 знаменатель Михайловой (диагностика)', 'код'),
 ('geometry_mode', '1 = вся длина канала; 0 = четыре участка AIR', 'код'),
 ('connection_count_override', 'Число соединений; 0 = оценка H/lt по AIR', ''),
 ('air_substeps', 'Число подучастков M1 (1 = формула AIR на весь участок)', ''),
]


def normalized(data):
    d = default_data()
    unknown = set(data)-set(d)
    if unknown:
        raise ValueError('Неизвестные поля: '+', '.join(sorted(unknown)))
    d.update(data)
    if any(type(v) not in (int, float) for v in d.values()):
        raise ValueError('Все исходные параметры должны быть числами.')
    require_finite(*d.values())
    positive = ('brhl_diam_D','dpth_h','out_diam_dril_pipe_d','in_diam_dril_pipe_d1',
                'diam_chnl_join_d3','pipe_lngth_lt','out_diam_core_pipe_dh','in_diam_core_pipe_db',
                'core_pipe_lngth_lk','core_diam_dk','rock_density_ro','well_temp_t','gast_koef_kr',
                'coef_spd_unvnss_k','coef_red_lift_frc_k1','rqrd_up_spd_v1','ATM_PRSSR_P0',
                'GAS_CONST_R','ACCLRTN_GRAV_g','flow_reference_temperature_k','compressor_intake_temperature_k')
    if min(d[k] for k in positive) <= 0:
        raise ValueError('Размеры, температуры и свойства должны быть положительными.')
    if not (0 <= d['drilling_angle_alfa'] <= 90 and 0 <= d['core_out_bk'] <= 100
            and d['mech_spd_vm'] >= 0 and 0 < d['comprsr_effency_ni'] <= 1
            and d['AIR_ADBTC_INDX_k'] > 1 and d['receiver_margin_pa'] >= 0
            and d['air_power_N0'] >= 0 and d['connection_count_override'] >= 0):
        raise ValueError('Недопустимые коэффициенты, выход керна, угол или проходка.')
    for key, valid in [('guo_friction_mode',(0,1,2)),('source_literal_mode',(0,1)),('geometry_mode',(0,1)),
                       ('up_formula_variant',(1,2)),('cnctn_type_j',(1,2,3)),('tool_type_j1',(1,2,3))]:
        if d[key] not in valid:
            raise ValueError('Некорректный код '+key)
    if not (d['dpth_h'] > d['core_pipe_lngth_lk']
            and d['brhl_diam_D'] > max(d['out_diam_dril_pipe_d'],d['out_diam_core_pipe_dh'])
            and d['out_diam_dril_pipe_d'] > d['in_diam_dril_pipe_d1'] >= d['diam_chnl_join_d3']
            and d['out_diam_core_pipe_dh'] > d['in_diam_core_pipe_db'] > d['core_diam_dk']):
        raise ValueError('Размеры каналов не согласованы.')
    if d['air_substeps'] != int(d['air_substeps']) or not 1<=d['air_substeps']<=4096:
        raise ValueError('Число подучастков должно быть целым от1 до4096.')
    if min(d[k] for k in ('pipe_roughness_m','borehole_roughness_m','core_roughness_m'))<0:
        raise ValueError('Шероховатость не может быть отрицательной.')
    if d['guo_friction_mode'] and d['source_literal_mode']:
        raise ValueError('Отключите буквальный AIR для отдельного закона трения Guo.')
    if d['guo_friction_mode']==1 and min(d[k] for k in ('pipe_roughness_m','borehole_roughness_m','core_roughness_m'))<=0:
        raise ValueError('Для Nikuradse задайте три положительные шероховатости; гладкая стенка требует другого закона трения.')
    return d


def project_common(data, q=None):
    d = normalized(data)
    if q is None:
        q = 47.1*d['coef_spd_unvnss_k']*d['coef_red_lift_frc_k1']*(d['brhl_diam_D']**2-d['out_diam_dril_pipe_d']**2)*d['rqrd_up_spd_v1']
    if type(q) not in (int,float):
        raise ValueError('Расход должен быть числом.')
    require_finite(q)
    if q <= 0:
        raise ValueError('Расход должен быть положительным.')
    literal = bool(d['source_literal_mode'])
    pi = 3.14159 if literal else math.pi
    # Literal AIR line410 uses well T despite its stated normal-volume basis.
    reference_T = d['well_temp_t'] if literal else d['flow_reference_temperature_k']
    gas = Gas(d['GAS_CONST_R'],d['AIR_ADBTC_INDX_k'],d['ACCLRTN_GRAV_g'],d['ATM_PRSSR_P0'],reference_T)
    mg = gas.mass_flow(q)
    ms = pi/4*(d['brhl_diam_D']**2-d['core_out_bk']/100*d['core_diam_dk']**2)*d['rock_density_ro']*d['mech_spd_vm']/3600
    return d, gas, q, mg, ms, pi


def project_channels(data):
    d, gas, q, mg, ms, pi = project_common(data)
    literal = bool(d['source_literal_mode'])
    closed = bool(d['geometry_mode']) and not literal
    L = d['core_pipe_lngth_lk']; H = d['dpth_h']; core_length = L*d['core_out_bk']/100
    D, dp, di, od, db, core = (d[k] for k in ('brhl_diam_D','out_diam_dril_pipe_d',
                                 'in_diam_dril_pipe_d1','out_diam_core_pipe_dh','in_diam_core_pipe_db','core_diam_dk'))
    # Calculation runs against the flow: outside down, then inside up.
    rows = [('annulus_drill_pipe',H-L,pi*(D*D-dp*dp)/4,D-dp,'up',0,H-L),
            ('annulus_core_barrel',L,pi*(D*D-od*od)/4,D-od,'up',H-L,H),
            ('around_core',core_length,pi*(db*db-core*core)/4,db-core,'down',H-core_length,H)]
    if closed:
        rows.append(('empty_core_barrel',L-core_length,pi*db*db/4,db,'down',H-L,H-core_length))
    rows.append(('inside_drill_pipe',H-L,pi*di*di/4,di,'down',0,H-L))
    exponent = .333 if literal else 1/3
    count = H/d['pipe_lngth_lt'] if literal or d['connection_count_override']==0 else d['connection_count_override']
    xi = {1:2.,2:1.5,3:0.}[d['cnctn_type_j']]*((di/d['diam_chnl_join_d3'])**2-1)**2
    channels = []
    for name, length, area, dh, flow, top, bottom in rows:
        if length == 0:
            continue
        f = .009407/dh**exponent
        le = di*count*xi/f if name == 'inside_drill_pipe' else 0.
        section = Section(name,length,area,dh,f, d['well_temp_t'],d['well_temp_t'],90-d['drilling_angle_alfa'])
        channels.append(dict(section=section,flow=flow,equivalent_length=le,
                             top_depth=top,bottom_depth=bottom))
    return channels, dict(connection_count=count,xi_each=xi,geometry_closed=closed,
                          omitted_empty_barrel_length_m=0. if closed else L-core_length)


def shared_power(d, gas, mg, receiver_pa):
    """Common single-stage ideal-gas estimate, not native AIR or Guo power.
    NASA Compressor Thermodynamics: cp*Tin*(ratio**((gamma-1)/gamma)-1)/eta.
    Intake and receiver pressures are approximated as stagnation pressures.
    No intercooling, stage selection, mechanical or motor efficiency model.
    """
    ratio = receiver_pa/d['ATM_PRSSR_P0']
    if ratio < 1:
        return None
    k = gas.k
    return mg*gas.R*d['compressor_intake_temperature_k']*k/(k-1)*math.expm1((k-1)/k*math.log(ratio))/d['comprsr_effency_ni']


def friction_regime_check(T,mg,area,dh,roughness):
    """Dry-air diagnostic. COMSOL Sutherland constants; Colebrook pipe analogy.
    5% is a project comparison tolerance, not an author validity boundary.
    """
    require_finite(T,mg,area,dh)
    if min(T,mg,area,dh)<=0:raise ValueError('Положительные температура, расход и геометрия обязательны.')
    if roughness is not None:
        require_finite(roughness)
        if roughness<0:raise ValueError('Отрицательная шероховатость.')
    if not 170<=T<=1900:return {'status':'temperature outside viscosity diagnostic'}
    viscosity=1.716e-5*(T/273)**1.5*(273+111)/(T+111)
    Re=mg*dh/(area*viscosity)
    result={'Re_gas':Re,'viscosity_Pa_s':viscosity,
            'status':'laminar' if Re<2300 else 'transition' if Re<4000 else 'turbulent',
            'scope':'gas-only, hydraulic-diameter approximation; not validation of gas-solids friction'}
    if roughness is None:return result
    relative=roughness/dh;result['relative_roughness']=relative
    if Re<4000 or relative>.1:
        result['comparison_status']='outside chosen Colebrook diagnostic range';return result
    lo,hi=.5,100.
    for _ in range(80):
        x=(lo+hi)/2
        if x+2*math.log10(relative/3.7+2.51*x/Re)<0:lo=x
        else:hi=x
    f=((lo+hi)/2)**-2
    result['colebrook_f']=f
    if roughness==0:
        result.update(comparison_status='smooth-wall Colebrook limit; fully-rough Nikuradse not applicable',nikuradse_f=None,nikuradse_vs_colebrook_pct=None,within_project_5pct_tolerance=None)
        return result
    fn=nikuradse(roughness,dh)
    delta=100*(fn/f-1)
    result.update(colebrook_f=f,nikuradse_f=fn,nikuradse_vs_colebrook_pct=delta,
                  within_project_5pct_tolerance=abs(delta)<=5,
                  viscous_to_rough_term=(2.51/(Re*math.sqrt(f)))/(relative/3.7))
    return result


def calculate(data, q=None, method='air'):
    if method not in ('air','guo'):
        raise ValueError('Метод air или guo.')
    d, gas, q, mg, ms, pi = project_common(data,q)
    channels, geometry = project_channels(d)
    pressure = d['ATM_PRSSR_P0']; checkpoints = {}; boundaries=[]
    b = gas.g*math.sin(d['drilling_angle_alfa']*pi/180)/(2*gas.R*d['well_temp_t'])
    mu = ms/mg
    plus = d['source_literal_mode'] or d['up_formula_variant']==1
    for row in channels:
        s = row['section']; up = row['flow']=='up'; p_in = pressure
        f = s.friction_darcy
        equivalent_length=row['equivalent_length']
        wall_roughness=None
        if method=='guo' and d['guo_friction_mode']:
            ep=d['pipe_roughness_m'];eh=d['borehole_roughness_m'];ec=d['core_roughness_m']
            # Eq6.20, Hc=0 for uncased project; local wetted-perimeter weighting.
            # Around-core weighting is a declared extension, not an author example.
            if s.name.startswith('annulus'):
                inner=d['out_diam_dril_pipe_d'] if s.name=='annulus_drill_pipe' else d['out_diam_core_pipe_dh']
                wall_roughness=(ep*inner+eh*d['brhl_diam_D'])/(inner+d['brhl_diam_D'])
            elif s.name=='around_core':
                wall_roughness=(ep*d['in_diam_core_pipe_db']+ec*d['core_diam_dk'])/(d['in_diam_core_pipe_db']+d['core_diam_dk'])
            else:wall_roughness=ep
            if d['guo_friction_mode']==2:
                diag=friction_regime_check(d['well_temp_t'],mg,s.area_m2,s.hydraulic_diameter_m,wall_roughness)
                if 'colebrook_f' not in diag:raise ValueError('Colebrook: требуется Re>=4000 и e/Dh<=0.1 в области диагностики вязкости.')
                f=diag['colebrook_f']
            else:f=nikuradse(wall_roughness,s.hydraulic_diameter_m)
            equivalent_length*=s.friction_darcy/f  # keep joint K fixed when wall friction changes
        if method=='air':
            a = mg*mg*gas.R*d['well_temp_t']/(s.hydraulic_diameter_m*s.area_m2**2)
            parts=1 if d['source_literal_mode'] else int(d['air_substeps'])
            length=s.length_m/parts;extra=row['equivalent_length']/parts
            for _ in range(parts):
                if up:
                    denominator=1-(1+mu if plus else 1-mu)*b*length
                    if denominator<=0:
                        raise ValueError('Знаменатель M1 неположителен: метод неприменим к участку.')
                    pressure=(math.sqrt(pressure**2+f*(1+d['gast_koef_kr']*mu)*a*length)+pressure*(1+mu)*b*length)/denominator
                else:
                    pressure=(math.sqrt(pressure**2+f*a*(length+extra))-pressure*b*length)/(1+b*length)
        else:
            # Spread joint K into friction only; gravitational length remains real L.
            effective=replace(s,friction_darcy=f*(1+equivalent_length/s.length_m))
            pressure=section_pressure(pressure,effective,gas,mg,ms if up else 0.,row['flow'],'top' if up else 'bottom')
        require_finite(pressure)
        if pressure<=0:
            raise ValueError('Неположительное абсолютное давление.')
        if s.name=='annulus_drill_pipe':checkpoints['P1']=pressure/1e6
        if s.name=='annulus_core_barrel':checkpoints['P2']=pressure/1e6
        if s.name=='around_core':checkpoints['P3']=pressure/1e6
        top_p,bottom_p=(p_in,pressure) if up else (pressure,p_in)
        states=[state(p,d['well_temp_t'],s.area_m2,mg,gas) for p in (top_p,bottom_p)]
        boundaries.append(dict(name=s.name,flow=row['flow'],length_m=s.length_m,
                               top_depth_m=row['top_depth'],bottom_depth_m=row['bottom_depth'],
                               area_m2=s.area_m2,dh_m=s.hydraulic_diameter_m,friction=f,roughness_m=wall_roughness,
                               friction_regime=friction_regime_check(d['well_temp_t'],mg,s.area_m2,s.hydraulic_diameter_m,wall_roughness),
                               equivalent_length_m=equivalent_length,top_pressure_pa=top_p,
                               bottom_pressure_pa=bottom_p,states=states))
    if 'P3' not in checkpoints:checkpoints['P3']=checkpoints['P2']  # no core, zero inner segment
    checkpoints['P4']=pressure/1e6
    pres=(pressure+d['receiver_margin_pa'])/1e6
    n_iso=shared_power(d,gas,mg,pres*1e6)
    n_air=11*d['air_power_N0']*q*pres*1000 if d['air_power_N0']>0 and method=='air' else None
    all_states=[v for r in boundaries for v in r['states']]
    annular_states=[v for r in boundaries if r['flow']=='up' for v in r['states']]
    minimum_kei=min(v['kinetic_energy_index'] for v in annular_states)
    max_mach=max(v['gas_velocity_m_s']/math.sqrt(gas.k*gas.R*d['well_temp_t']) for v in all_states)
    max_solids=max((ms/(d['rock_density_ro']*(mg/v['density_kg_m3']+ms/d['rock_density_ro']))) for r in boundaries if r['flow']=='up' for v in r['states'])
    warnings=['Reynolds оценён для газовой фазы; это не проверка режима газошламовой смеси.',
              'Скольжение шлама, ускорение в переходах, потери в коронке, утечки и теплообмен не моделируются.',
              'Сопротивления соединений сохранены как проектное допущение; шероховатость керна не подтверждена измерениями.']
    if d['air_substeps']>1 and not d['source_literal_mode']:warnings.append('M1 разбит на подучастки: это проверка сходимости, а не буквальный расчёт AIR одним шагом.')
    if d['source_literal_mode']:warnings.append('Буквальный AIR: массовая подача рассчитана при T скважины; это не согласуется с обозначением Q как расхода при273K в тексте источника.')
    if not geometry['geometry_closed']:warnings.append('По схеме AIR пропущено %.6g м пустого участка колонковой трубы.'%geometry['omitted_empty_barrel_length_m'])
    if d['up_formula_variant']==2 and not d['source_literal_mode']:warnings.append('Выбран только вариант знаменателя VI.26; это не полный расчёт по Михайловой.')
    if any(r['friction_regime'].get('within_project_5pct_tolerance') is False for r in boundaries):warnings.append('Nikuradse отличается от Colebrook более чем на проектный допуск 5% хотя бы в одном канале; независимость трения от Reynolds не подтверждена.')
    if method=='guo' and d['guo_friction_mode']==2:warnings.append('Выбрана модификация Guo с Colebrook, а не авторский закон Nikuradse. Кольцевые каналы приближены гидравлическим диаметром.')
    if method=='air' or d['guo_friction_mode']==0:warnings.append('Используется закон Weymouth AIR; наличие газового Reynolds само по себе не валидирует эту корреляцию.')
    if method=='guo' and d['guo_friction_mode']==1:warnings.append('Guo: Nikuradse 6.19; веса стенок по 6.20 при Hc=0. Для керна это перенос правила. Местные потери соединений сохранены как проектное допущение. Проверка Reynolds является газовой диагностикой; полевой пример не подтверждён.')
    if ms==0:warnings.append('Шлам не генерируется: KEI выведен справочно, задача выноса в этом режиме не проверяется.')
    elif minimum_kei<1:warnings.append('Минимальный индекс кинетической энергии в затрубье <1: базовый критерий Angel не выполнен; вынос шлама не подтверждён.')
    else:warnings.append('Индекс энергии >=1 выполняет только базовый критерий Angel; это не гарантия выноса частиц заданного размера.')
    if d['drilling_angle_alfa']<90:warnings.append('Наклонная геометрия рассчитана гидравлически; энергетический критерий не подтверждает отсутствие осевшего слоя шлама.')
    if max_mach>=1:warnings.append('На границах Ma>=1: численный ответ не подтверждает физическую применимость дозвуковой модели.')
    if max_mach>=.3:warnings.append('На границах участков Ma>=0.3: влияние ускорения газа требует отдельной проверки.')
    return {**checkpoints,'Pres':pres,'N':n_air if d['source_literal_mode'] else n_iso,
            'N_isentropic_W':n_iso,'N_AIR_table_W':n_air,
            'vkol':q/(60*(pi/4*(d['brhl_diam_D']**2-d['out_diam_dril_pipe_d']**2))),
            'vbt':q/(60*(pi/4)*d['in_diam_dril_pipe_d1']**2),
            'Q_reference_m3_min':q,'mass_gas_kg_s':mg,'mass_solids_kg_s':ms,
            'effective_reference_temperature_K':gas.reference_temperature_k,
            'minimum_annular_endpoint_KEI':minimum_kei,
            'maximum_endpoint_Mach':max_mach,'homogeneous_solids_volume_fraction_max':max_solids,
            'geometry':geometry,'sections':boundaries,'warnings':warnings,
            'method':method,'settings':d,
            'effective_settings':dict(up_variant=1 if d['source_literal_mode'] else d['up_formula_variant'],
                                      air_substeps=1 if d['source_literal_mode'] else d['air_substeps'],
                                      friction_exponent=.333 if d['source_literal_mode'] else 1/3)}


def hydraulic_model(data, G):
    """Compatibility: G is reference m3/min, never kg/s. N is W or None."""
    return calculate(data,G,'air')


def hydraulic_model_m2(data, G):
    return calculate(data,G,'guo')


def calculate_pair(data, flow=None):
    q=project_common(data,flow)[2]
    return q,calculate(data,q,'air'),calculate(data,q,'guo')


def show_results(data, flow=None):
    q,a,b=calculate_pair(data,flow)
    print('\nAIR и Guo, общий расход %.9g м³/мин; массовая подача %.9g кг/с'%(q,a['mass_gas_kg_s']))
    print('Режим: '+('буквальный AIR (контроль источника)' if a['settings']['source_literal_mode'] else 'единые исходные условия'))
    print('Давления абсолютные. Мощность по общей одноступенчатой оценке; не подбор компрессора.')
    print('%-20s %14s %14s %12s'%('Показатель','AIR','Guo адапт.','Разница,%'))
    for key in ('P1','P2','P3','P4','Pres','N','vkol','vbt'):
        x,y=a[key],b[key];scale=1000 if key=='N' else 1
        sx='не определена' if x is None else '%.6f'%(x/scale)
        sy='не определена' if y is None else '%.6f'%(y/scale)
        delta='-' if key=='N' or x in (None,0) or y is None else '%.4f'%(100*(y/x-1))
        print('%-20s %14s %14s %12s'%(key+(' кВт' if key=='N' else ' МПа' if key.startswith('P') else ' м/с'),sx,sy,delta))
    if a['settings']['source_literal_mode']:
        print('N в столбце AIR требует N0; столбец Guo не имеет независимой книжной мощности.')
    for row in b['sections']:
        diag=row['friction_regime'];print('Трение %s: Re=%s, Nikuradse/Colebrook, %%=%s'%(row['name'],diag.get('Re_gas','н/д'),diag.get('nikuradse_vs_colebrook_pct','н/д')))
    print('Минимальный индекс энергии в затрубье: AIR %.6f; Guo %.6f (базовый порог 1).'%(a['minimum_annular_endpoint_KEI'],b['minimum_annular_endpoint_KEI']))
    print('P3 находится у верхнего конца керна; при замкнутой геометрии далее учтён пустой участок трубы.')
    print('vkol и vbt рассчитаны по приведённому расходу; местные скорости есть в экспорте.')
    for warning in dict.fromkeys(a['warnings']+b['warnings']):print('Примечание: '+warning)


def research(data, destination):
    rows=[]
    for H in (100.,200.,300.):
        for v in (3.,6.,9.):
            for q in (3.,project_common(data)[2],6.):
                d=dict(data,dpth_h=H,mech_spd_vm=v)
                a,b=calculate(d,q,'air'),calculate(d,q,'guo')
                rows.append(dict(H_m=H,penetration_m_h=v,Q_m3_min=q,AIR_P4_MPa=a['P4'],Guo_P4_MPa=b['P4'],
                                 difference_percent=100*(b['P4']/a['P4']-1),maximum_endpoint_Mach=max(a['maximum_endpoint_Mach'],b['maximum_endpoint_Mach'])))
    with Path(destination).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows



def refinement_study(data):
    """Test the common differential limit, not fit an author answer.
    For AIR sign and kr=1, small segments tend to the same dry pressure ODE.
    """
    d=normalized(data)
    d.update(source_literal_mode=0,up_formula_variant=1,gast_koef_kr=1.,guo_friction_mode=0)
    q=project_common(d)[2];target=calculate(d,q,'guo')['P4'];rows=[]
    for n in (1,2,4,8,16,32,64,128,256,512,1024):
        r=calculate(dict(d,air_substeps=n),q,'air')
        rows.append(dict(substeps=n,AIR_P4_MPa=r['P4'],Guo_P4_MPa=target,
                         relative_difference_percent=100*(r['P4']/target-1)))
    return rows


def target_flow(data, method, parameter, target, qmin, qmax, tolerance=1e-8):
    """Find a bracketed target, not a global optimum or a monotonicity proof.
    Q units m3/min; pressures MPa absolute; power W; velocities reference m/s.
    """
    if parameter not in ('P1','P2','P3','P4','Pres','N','vkol','vbt'):
        raise ValueError('Неизвестная расчётная цель.')
    require_finite(target,qmin,qmax,tolerance)
    if not 0<qmin<qmax or tolerance<=0:raise ValueError('Неверный интервал или допуск.')
    def evaluate(q):
        r=calculate(data,q,method);value=r[parameter]
        if value is None:raise ValueError('Целевой показатель не определён в этом режиме.')
        return value-target,r
    left,_=evaluate(qmin);right,_=evaluate(qmax)
    if abs(left)<=tolerance:return dict(Q_m3_min=qmin,iterations=0,result=calculate(data,qmin,method))
    if abs(right)<=tolerance:return dict(Q_m3_min=qmax,iterations=0,result=calculate(data,qmax,method))
    if left*right>0:raise ValueError('На концах интервала нет смены знака невязки.')
    for i in range(100):
        mid=(qmin+qmax)/2;error,r=evaluate(mid)
        if abs(error)<=tolerance:return dict(Q_m3_min=mid,iterations=i+1,result=r)
        if left*error<0:qmax=mid
        else:qmin=mid;left=error
    raise ValueError('Допуск цели не достигнут за100 итераций.')


def flow_scan(data, qmin, qmax, count, destination):
    if not 0<qmin<qmax or type(count) is not int or not 2<=count<=10000:
        raise ValueError('Нужен положительный интервал и от2 до10000 точек.')
    rows=[]
    for i in range(count):
        q=qmin+(qmax-qmin)*i/(count-1)
        try:
            _,a,b=calculate_pair(data,q)
            rows.append(dict(Q_m3_min=q,AIR_P4_MPa=a['P4'],Guo_P4_MPa=b['P4'],
                             AIR_power_W=a['N'],Guo_common_power_W=b['N'],status='calculated',
                             diagnostic='; '.join(dict.fromkeys(a['warnings']+b['warnings']))))
        except (ValueError,ArithmeticError) as exc:
            rows.append(dict(Q_m3_min=q,AIR_P4_MPa=None,Guo_P4_MPa=None,AIR_power_W=None,
                             Guo_common_power_W=None,status='rejected',diagnostic=str(exc)))
    with Path(destination).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def number(raw):
    v=float(raw.strip().replace(',','.'));require_finite(v);return v


def main():
    data=default_data();flow=None
    print('СУХАЯ ПРОДУВКА: М1 AIR, М2 Guo и М3 v1.2. Все вычисления в одном файле.')
    while True:
        print('\n1. Исходные данные\n2. Изменить параметр\n3. Общий расход\n4. Сравнить модели\n'
              '5. Контроль книжного AIR\n6. Сохранить исходные данные\n7. Загрузить исходные данные\n'
              '8. Экспорт расчёта JSON\n9. Исследование 27 случаев CSV\n10. Встроенные проверки\n11. Сходимость уравнений при kr=1\n12. Подбор расхода по цели\n13. Сканирование расхода CSV\n14. Прямой анализ расчётов\n15. Сохранить анализ (HTML, JSON, CSV)\n16. Анализ диапазона расхода\n17. М3: раздельные скорости газа и шлама\n18. Сравнение М1/М2/М3 на отдельных участках\n19. Аудит нормировки трения М3\n20. М3: весь контур, паспорт, анализ и экспорт\n21. М3: экспериментальные контроли и неопределённость\n22. Контроль численной точности М3\n0. Выход')
        try:
            choice=input('Пункт: ').strip()
            if choice=='0':return
            if choice=='22':
                m3_numerical_quality_menu(data,flow);continue
            if choice=='21':
                m3_evidence_menu(data,flow);continue
            if choice=='20':
                m3_full_menu(data,flow);continue
            if choice=='19':
                m3_source_audit_menu(data,flow);continue
            if choice=='18':
                three_model_menu(data,flow);continue
            if choice=='17':
                m3_menu(data,flow);continue
            if choice=='1':
                for i,(key,label,unit) in enumerate(FIELDS,1):print('%2d. %s: %g %s'%(i,label,data[key],unit))
            elif choice=='2':
                for i,(key,label,unit) in enumerate(FIELDS,1):print('%2d. %s: %g %s'%(i,label,data[key],unit))
                i=int(input('Номер: '))-1
                if not 0<=i<len(FIELDS):raise ValueError('Нет параметра.')
                key,label,unit=FIELDS[i];candidate=dict(data);candidate[key]=number(input(label+' '+unit+': '))
                data=normalized(candidate)
            elif choice=='3':
                raw=input('Расход м³/мин; a = по формуле47.1: ').strip()
                candidate=None if raw.lower()=='a' else number(raw)
                project_common(data,candidate);flow=candidate
            elif choice=='4':show_results(data,flow)
            elif choice=='5':
                d=default_data();d['source_literal_mode']=1
                show_results(d)
                print('Печатный контроль: Q=4.58755 м³/мин, Pres=0.67253 МПа. N0 не назначен.')
            elif choice in ('6','7','8','9'):
                default='inputs.json' if choice=='7' else unique_output(Path(__file__).resolve().parent,{'6':'inputs','8':'results','9':'scenarios'}[choice],'csv' if choice=='9' else 'json').name
                raw=input('Файл ['+default+']: ').strip().strip('"')
                path=Path(raw).expanduser() if raw else Path(__file__).resolve().with_name(default)
                if choice=='6':path.write_text(json.dumps(dict(schema_version=2,inputs=data,flow_override_m3_min=flow),ensure_ascii=False,indent=2),encoding='utf-8')
                elif choice=='7':
                    payload=json.loads(path.read_text(encoding='utf-8-sig'))
                    if not isinstance(payload,dict):raise ValueError('Ожидается объект JSON с исходными данными.')
                    if payload.get('schema_version') not in (1,2):raise ValueError('Неверная версия файла.')
                    candidate=normalized(payload['inputs']);qnew=payload.get('flow_override_m3_min')
                    project_common(candidate,qnew);data,flow=candidate,qnew
                elif choice=='8':
                    q,a,b=calculate_pair(data,flow)
                    path.write_text(json.dumps(dict(schema_version=2,units=dict(pressure='MPa absolute',power='W',Q='reference m3/min'),AIR=a,Guo_adapted=b),ensure_ascii=False,indent=2),encoding='utf-8')
                else:research(data,path)
                print('Готово: '+str(path.resolve()))
            elif choice=='12':
                method=input('Метод air/guo: ').strip()
                parameter=input('Цель P1/P2/P3/P4/Pres/N/vkol/vbt (N в Вт): ').strip()
                target=number(input('Значение: '));lo=number(input('Минимальный Q, м³/мин: '));hi=number(input('Максимальный Q: '))
                result=target_flow(data,method,parameter,target,lo,hi)
                print('Q=%.9g м³/мин, итераций %d'%(result['Q_m3_min'],result['iterations']))
                print('Показатель: ',result['result'][parameter])
                for warning in result['result']['warnings']:print('Примечание: '+warning)
            elif choice=='13':
                lo=number(input('Минимальный Q, м³/мин: '));hi=number(input('Максимальный Q: '));count=int(input('Число точек: '))
                raw=input('CSV-файл [автоматическое имя с датой]: ').strip().strip(chr(34))
                path=Path(raw).expanduser() if raw else unique_output(Path(__file__).resolve().parent,'flow_scan','csv')
                flow_scan(data,lo,hi,count,path);print('Сохранено: '+str(path.resolve()))
            elif choice=='14':
                print(analysis_text(analyze_models(data,flow)))
            elif choice=='15':
                raw=input('Папка отчёта [рядом с программой]: ').strip().strip(chr(34))
                folder=Path(raw).expanduser() if raw else Path(__file__).resolve().parent
                for path in export_analysis(analyze_models(data,flow),folder):print('Сохранено: '+str(path))
            elif choice=='16':
                lo=number(input('Минимальный Q, м³/мин: '));hi=number(input('Максимальный Q: '));count=int(input('Число точек (2..1001): '))
                scan=analyze_flow_range(data,lo,hi,count)
                print(json.dumps(scan['summary'],ensure_ascii=False,indent=2))
                path=unique_output(Path(__file__).resolve().parent,'flow_analysis','json')
                path.write_text(json.dumps(scan,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print('Сохранено: '+str(path))
            elif choice=='11':
                for row in refinement_study(data):print(row)
            elif choice=='10':unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Verification))
            else:print('Нет такого пункта.')
        except (EOFError,KeyboardInterrupt):return
        except (ValueError,TypeError,KeyError,ArithmeticError,OSError) as e:print('Ошибка: '+str(e))


class Verification(unittest.TestCase):
    def test_reference_pipe_velocity(self):
        r=calculate(default_data(),4.5)
        area=math.pi*.039**2/4
        self.assertAlmostEqual(r['vbt']*area*60,4.5,places=12)
        inner=r['sections'][-1]
        for v in inner['states']:
            self.assertAlmostEqual(v['density_kg_m3']*v['gas_velocity_m_s']*inner['area_m2'],r['mass_gas_kg_s'],places=12)
    def test_low_energy_warning(self):
        r=calculate(default_data(),.5)
        self.assertLess(r['minimum_annular_endpoint_KEI'],1)
        self.assertTrue(any('критерий Angel не выполнен' in w for w in r['warnings']))
    def test_regime_diagnostic(self):
        r=friction_regime_check(273, .1, .001, .04, .00004)
        self.assertAlmostEqual(r['viscosity_Pa_s'],1.716e-5,places=12)
        self.assertEqual(r['status'],'turbulent')
        x=1/math.sqrt(r['colebrook_f'])
        self.assertAlmostEqual(x+2*math.log10(.001/3.7+2.51*x/r['Re_gas']),0,places=12)
        low=friction_regime_check(290, .000001, .001, .04, .00004)
        self.assertEqual(low['status'],'laminar')
    def test_nikuradse_mode(self):
        d=dict(default_data(),guo_friction_mode=1,pipe_roughness_m=.00004572,borehole_roughness_m=.0001,core_roughness_m=.0001)
        a=calculate(d,method='guo');b=calculate(default_data(),method='guo')
        self.assertNotEqual(a['P4'],b['P4'])
        for row in a['sections']:
            self.assertAlmostEqual(row['friction'],nikuradse(row['roughness_m'],row['dh_m']),places=14)
        old=calculate(default_data(),method='guo')['sections'][-1];new=a['sections'][-1]
        self.assertAlmostEqual(old['friction']*old['equivalent_length_m'],new['friction']*new['equivalent_length_m'],places=12)
        with self.assertRaises(ValueError):calculate(dict(default_data(),guo_friction_mode=1),method='guo')
        with self.assertRaises(ValueError):calculate(dict(d,borehole_roughness_m=.00762),method='guo')
    def test_published_air(self):
        d=default_data();d['source_literal_mode']=1;r=calculate(d)
        self.assertLess(abs(r['Pres']-.67253),.000005)
        self.assertAlmostEqual(r['Pres'],.6725305051146977,places=12)
        self.assertIsNone(r['N'])
    def test_common_mass_and_geometry(self):
        q,a,b=calculate_pair(default_data())
        self.assertEqual(a['mass_gas_kg_s'],b['mass_gas_kg_s'])
        self.assertAlmostEqual(a['mass_gas_kg_s'],q*1e5/(60*287*273),places=14)
        for r in (a,b):
            self.assertAlmostEqual(sum(s['length_m'] for s in r['sections'] if s['flow']=='down'),300)
            self.assertAlmostEqual(sum(s['length_m'] for s in r['sections'] if s['flow']=='up'),300)
    def test_zero_full_core(self):
        for recovery in (0,100):
            _,a,b=calculate_pair(dict(default_data(),core_out_bk=recovery))
            self.assertGreater(a['P4'],0);self.assertGreater(b['P4'],0)
    def test_ideal_reference_invariance(self):
        d=default_data();q=project_common(d)[2];a=calculate(d,q)
        d['flow_reference_temperature_k']=300.;b=calculate(d,q*300/273)
        self.assertAlmostEqual(a['P4'],b['P4'],places=12)
        self.assertAlmostEqual(a['N'],b['N'],places=8)
    def test_horizontal_no_solids_exact(self):
        d=dict(default_data(),drilling_angle_alfa=0,mech_spd_vm=0)
        _,a,b=calculate_pair(d)
        for k in ('P1','P2','P3','P4'):self.assertAlmostEqual(a[k],b[k],places=12)
    def test_power_uses_gamma_and_external_N0(self):
        d=default_data();r=calculate(d);mg=r['mass_gas_kg_s'];k=1.4
        expected=mg*287*293.15*k/(k-1)*((r['Pres']*1e6/1e5)**((k-1)/k)-1)/.3
        self.assertAlmostEqual(r['N'],expected,places=8)
        d['source_literal_mode']=1;d['air_power_N0']=.98;r=calculate(d)
        self.assertAlmostEqual(r['N']/1000,33.25916383230421,places=10)
    def test_guo_against_rk4(self):
        d=default_data();r=calculate(d,method='guo');_,gas,q,mg,ms,pi=project_common(d)
        for row,record in zip(project_channels(d)[0],r['sections']):
            s=row['section'];up=row['flow']=='up';M=1+ms/mg if up else 1
            f=s.friction_darcy*(1+row['equivalent_length']/s.length_m)
            cosine=math.cos(math.radians(s.inclination_from_vertical_deg))
            def derivative(p):
                rho=p/(gas.R*d['well_temp_t']);v=mg/(rho*s.area_m2)
                return M*rho*(gas.g*cosine+(1 if up else -1)*f*v*v/(2*s.hydraulic_diameter_m))
            p=record['top_pressure_pa'] if up else record['bottom_pressure_pa'];dx=s.length_m/2000*(1 if up else -1)
            for _ in range(2000):
                a=derivative(p);b=derivative(p+dx*a/2);c=derivative(p+dx*b/2);e=derivative(p+dx*c)
                p+=dx*(a+2*b+2*c+e)/6
            expected=record['bottom_pressure_pa'] if up else record['top_pressure_pa']
            self.assertLess(abs(p/expected-1),1e-9)
    def test_common_differential_limit(self):
        rows=refinement_study(default_data())
        self.assertLess(abs(rows[-1]['relative_difference_percent']),abs(rows[0]['relative_difference_percent'])/100)
        self.assertLess(abs(rows[-1]['relative_difference_percent']),.001)
    def test_target_recovery(self):
        d=default_data();q=4.5;r=calculate(d,q,'guo')
        answer=target_flow(d,'guo','P4',r['P4'],2.,7.)
        self.assertAlmostEqual(answer['Q_m3_min'],q,places=6)
    def test_invalid_and_singular_inputs(self):
        for key,value in [('well_temp_t',math.nan),('core_out_bk',101),('source_literal_mode',2),('in_diam_core_pipe_db',.07)]:
            with self.assertRaises(ValueError):calculate(dict(default_data(),**{key:value}))
        with self.assertRaises(ValueError):calculate(default_data(),0)
        with self.assertRaises(ValueError):calculate(dict(default_data(),dpth_h=30000))


MODEL_LABELS = {'air':'AIR (источник ЛГИ, 1990)', 'guo_shared':'Guo: общее трение AIR',
                'guo_nikuradse':'Guo: Nikuradse, адаптация геометрии',
                'guo_colebrook':'Guo: Colebrook, исследовательская модификация'}
CHANNEL_LABELS = {'annulus_drill_pipe':'Затрубье бурильных труб','annulus_core_barrel':'Вокруг колонковой трубы',
                  'around_core':'Вокруг керна','empty_core_barrel':'Пустой участок колонковой трубы','inside_drill_pipe':'Внутри бурильных труб'}
SOURCE_REGISTER = [
 {'id':'TEMPLE1996','source':'Temple, Watson, Adewumi. Experimental study of pneumatic transport of solids in a vertical wellbore annulus, 1996, SPE 37327.',
  'url':'https://pure.psu.edu/en/publications/experimental-study-of-pneumatic-transport-of-solids-in-a-vertical/',
  'evidence':'Прочитана институциональная аннотация эксперимента: минимум перепада, влияние размера частиц, накопление при малой скорости. Численная валидация по таблицам не выполнена.'},
 {'id':'ADEWUMI1989','source':'Adewumi, Tian. Hydrodynamic modeling of wellbore hydraulics in air drilling, SPE 19333, 1989.',
  'url':'https://pure.psu.edu/en/publications/hydrodynamic-modeling-of-wellbore-hydraulics-in-air-drilling/',
  'evidence':'Институциональная аннотация: отдельная гидродинамическая модель транспорта частиц. Полный алгоритм здесь не восстановлен.'},
 {'id':'AIR1990','source':'Горшков, Климов, Козлов, Яковлев. Проектирование процессов бурения скважин с применением ЭВМ, 1990, с.27-36.',
  'url':'https://spmi.ru/sites/default/files/imci_images/univer/svedenia_jb_organizacii/metrek_spec/21.05.03-burenie-skvazhin.pdf',
  'evidence':'Листинг AIR и контроль прочитаны по сканам. Ссылка подтверждает библиографию, не заменяет скан формул.'},
 {'id':'MIKH1985','source':'Михайлова Н.Д. Техническое проектирование колонкового бурения, 1985, с.94-98.',
  'url':'https://www.geokniga.org/bookfiles/geokniga-tehnicheskoeproektirovaniekolonkovogobureniya.pdf',
  'evidence':'Скан: отличия VI.26 и VI.30 от AIR; единственный физически верный вариант не установлен.'},
 {'id':'GUO2011','source':'Guo B., Liu G. Applied Drilling Circulation Systems, Ch6, 2011.',
  'url':'https://doi.org/10.1016/B978-0-12-381957-4.00006-1',
  'evidence':'Книга прочитана. ОДУ давления, 6.19-6.20, KEI. Пример 6.4 и GasDrill-09.xls не воспроизведены.'},
 {'id':'LI2018','source':'Li, Yang, Guo, Liu. New Development of Air and Gas Drilling Technology, 2018.',
  'url':'https://www.intechopen.com/chapters/60339',
  'evidence':'Полный текст: критерии выноса и ограничения базового энергетического подхода.'},
 {'id':'SHAO2022','source':'Shao et al. Calculation of the Gas Injection Rate and Pipe String Erosion in Nitrogen Drilling Systems, 2022.',
  'url':'https://www.techscience.com/fdmp/v18n2/46011/html',
  'evidence':'Полевые данные для иных скважин; точность авторов нельзя присваивать нашему коду. Полная независимая репликация не выполнена.'},
 {'id':'COLEBROOK2018','source':'Praks, Brkic. Advanced iterative procedures for solving the implicit Colebrook equation, 2018.',
  'url':'https://arxiv.org/abs/1807.09913','evidence':'Формула трения при конечном Reynolds; перенос на кольцевой канал здесь приближённый.'},
 {'id':'ASHRAE','source':'ASHRAE Handbook, Duct Design, Table 1.',
  'url':'https://handbook.ashrae.org/Handbooks/F25/IP/F25_Ch21/F25_Ch21_ip.aspx',
  'evidence':'0.00015 ft для commercial steel. Справочное допущение, не измерение бурильной трубы.'},
 {'id':'SUTHERLAND','source':'COMSOL, Sutherland’s Law, параметры воздуха.',
  'url':'https://doc.comsol.com/6.4/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.46.html',
  'evidence':'Вязкость газовой фазы; не вязкость газошламовой смеси.'},
 {'id':'QUARMBY1967','source':'Quarmby A. An experimental study of turbulent flow through concentric annuli, 1967.',
  'url':'https://doi.org/10.1016/0020-7403(67)90030-6',
  'evidence':'Прочитана аннотация издателя: гладкие воздушные кольцевые каналы, Re=6000..90000. Не валидация наших шероховатых узких зазоров.'},
 {'id':'NASA_POWER','source':'NASA, Compressor Thermodynamics.',
  'url':'https://www.grc.nasa.gov/www/k-12/airplane/compth.html',
  'evidence':'Общая оценка изоэнтропической мощности; фактические характеристики компрессора не известны.'}
]


def timestamp():
    return datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d_%H-%M-%S')


def unique_output(folder, name, extension):
    folder=Path(folder);base=name+'_'+timestamp();path=folder/(base+'.'+extension);i=1
    while path.exists():
        path=folder/(base+'_'+str(i)+'.'+extension);i+=1
    return path


def result_diagnostics(r):
    notes=[];d=r['settings']
    if r['mass_solids_kg_s']==0:notes.append('Шлам не генерируется; KEI справочный.')
    elif r['minimum_annular_endpoint_KEI']<1:notes.append('Базовый критерий KEI>=1 не выполнен.')
    else:notes.append('KEI>=1 выполнен, но транспорт заданных частиц не подтверждён.')
    if r['maximum_endpoint_Mach']>=1:notes.append('Ma>=1: дозвуковое применение не подтверждено.')
    elif r['maximum_endpoint_Mach']>=.3:notes.append('Ma>=0.3: нужно проверить влияние ускорения; это диагностический порог.')
    reynolds=[s['friction_regime'].get('Re_gas') for s in r['sections']]
    if any(v is None for v in reynolds):notes.append('Не для всех участков получена диагностика вязкости и Reynolds.')
    elif min(reynolds)<4000:notes.append('Есть Re<4000: используемый турбулентный закон требует пересмотра.')
    if d['drilling_angle_alfa']<90:notes.append('Осевший слой в наклонном канале не моделируется.')
    if d['source_literal_mode']:notes.append('Буквальный книжный режим AIR; массовый расход рассчитан при T скважины.')
    if r['method']=='guo' and d['guo_friction_mode']==1:
        if any(s['friction_regime'].get('within_project_5pct_tolerance') is False for s in r['sections']):notes.append('Есть отличие Nikuradse/Colebrook >5% по f; полностью шероховатый предел не подтверждён для всего контура.')
    return notes


def analyze_models(data, flow=None):
    d,gas,q,mg,ms,pi=project_common(data,flow)
    models={};failures={}
    for key,method,mode in [('air','air',0),('guo_shared','guo',0),('guo_nikuradse','guo',1),('guo_colebrook','guo',2)]:
        try:
            candidate=dict(d,guo_friction_mode=mode)
            models[key]=calculate(candidate,q,method)
        except (ValueError,ArithmeticError) as exc:failures[key]=str(exc)
    if 'air' not in models:raise ValueError('AIR не рассчитан: '+failures.get('air','неизвестная причина'))
    air=models['air'];summary=[];sections=[]
    for key,r in models.items():
        summary.append(dict(model=key,label=MODEL_LABELS[key],P1_MPa=r['P1'],P2_MPa=r['P2'],P3_MPa=r['P3'],P4_MPa=r['P4'],
                            P4_gauge_MPa=r['P4']-d['ATM_PRSSR_P0']/1e6,Pres_MPa=r['Pres'],
                            delta_P4_vs_AIR_pct=100*(r['P4']/air['P4']-1),
                            common_isentropic_kW=None if r['N_isentropic_W'] is None else r['N_isentropic_W']/1000,
                            KEI_min=r['minimum_annular_endpoint_KEI'],Mach_max=r['maximum_endpoint_Mach'],notes=result_diagnostics(r)))
        for s in r['sections']:
            diag=s['friction_regime'];pin=s['top_pressure_pa'] if s['flow']=='up' else s['bottom_pressure_pa'];pout=s['bottom_pressure_pa'] if s['flow']=='up' else s['top_pressure_pa']
            sections.append(dict(model=key,channel=s['name'],label=CHANNEL_LABELS[s['name']],flow=s['flow'],
                                 p_top_MPa=s['top_pressure_pa']/1e6,p_bottom_MPa=s['bottom_pressure_pa']/1e6,
                                 reverse_path_increment_MPa=(pout-pin)/1e6,length_m=s['length_m'],
                                 friction_darcy=s['friction'],Re_gas=diag.get('Re_gas'),
                                 v_top_m_s=s['states'][0]['gas_velocity_m_s'],v_bottom_m_s=s['states'][1]['gas_velocity_m_s'],
                                 KEI_top=s['states'][0]['kinetic_energy_index'],KEI_bottom=s['states'][1]['kinetic_energy_index'],
                                 nikuradse_vs_colebrook_pct=diag.get('nikuradse_vs_colebrook_pct')))
    decomposition=None
    if not d['source_literal_mode'] and d['up_formula_variant']==1 and 'guo_shared' in models:
        ref=calculate(dict(d,guo_friction_mode=0,gast_koef_kr=1),q,'air')['P4']
        parts={'AIR_kr_effect_MPa':air['P4']-ref,'AIR_finite_step_effect_MPa':ref-models['guo_shared']['P4']}
        decomposition=dict(parts=parts,total_AIR_minus_Guo_shared_MPa=air['P4']-models['guo_shared']['P4'],
                           note='Последовательное алгебраическое разложение при одном Q; не независимая причинная оценка.')
    friction_effects={}
    if 'guo_shared' in models:
        for key in ('guo_nikuradse','guo_colebrook'):
            if key in models:friction_effects[key+'_minus_shared_MPa']=models[key]['P4']-models['guo_shared']['P4']
    if all(k in models for k in ('guo_nikuradse','guo_colebrook')):
        friction_effects['Colebrook_minus_Nikuradse_MPa']=models['guo_colebrook']['P4']-models['guo_nikuradse']['P4']
    return dict(schema_version=1,created_moscow=timestamp(),inputs=d,Q_reference_m3_min=q,mass_gas_kg_s=mg,mass_solids_kg_s=ms,
                requested_guo_mode=d['guo_friction_mode'],summary=summary,sections=sections,unavailable=failures,
                decomposition=decomposition,friction_effects=friction_effects,raw_results=models,sources=SOURCE_REGISTER,
                conclusion='Вычислительное сопоставление выполнено для доступных вариантов. Физическая точность и вынос шлама требуют независимой проверки.')


def analysis_text(a):
    lines=['ПРЯМОЙ АНАЛИЗ МОДЕЛЕЙ',
           'Q=%.9g м³/мин; mg=%.9g кг/с; ms=%.9g кг/с'%(a['Q_reference_m3_min'],a['mass_gas_kg_s'],a['mass_solids_kg_s']),
           'Давления абсолютные; мощность общая изоэнтропическая оценка.',
           'Модель | P4, МПа | разность с AIR, % | KEI min | Ma max']
    for r in a['summary']:
        lines.append('%s | %.6f | %+.4f | %.4f | %.4f'%(r['label'],r['P4_MPa'],r['delta_P4_vs_AIR_pct'],r['KEI_min'],r['Mach_max']))
        lines.extend('  '+n for n in r['notes'])
    lines.append('\nДАВЛЕНИЯ P1 / P2 / P3 / P4, МПа абсолютных')
    for r in a['summary']:lines.append('%s: %.6f / %.6f / %.6f / %.6f'%(r['label'],r['P1_MPa'],r['P2_MPa'],r['P3_MPa'],r['P4_MPa']))
    lines.append('\nУЧАСТКИ: приращения давления при обходе против потока, не чистые потери трения.')
    for key in a['raw_results']:
        lines.append(MODEL_LABELS[key])
        for r in a['sections']:
            if r['model']==key:lines.append('  %s: dP=%+.6f МПа; v верх/низ=%.3f/%.3f м/с; Re=%s'%(r['label'],r['reverse_path_increment_MPa'],r['v_top_m_s'],r['v_bottom_m_s'],'н/д' if r['Re_gas'] is None else '%.0f'%r['Re_gas']))
    if a['decomposition']:
        lines.append('\nРАЗЛОЖЕНИЕ AIR - Guo при общем трении')
        lines.extend('%s: %+.9f МПа'%(k,v) for k,v in a['decomposition']['parts'].items())
        lines.append(a['decomposition']['note'])
    lines.extend('%s: %+.9f МПа'%(k,v) for k,v in a['friction_effects'].items())
    for key,error in a['unavailable'].items():lines.append('Недоступно '+MODEL_LABELS[key]+': '+error)
    lines.append('Минимизация давления по Q не является проверенным подбором продувки: накопление частиц при малой скорости здесь не описано.')
    lines.extend(['\n'+a['conclusion'],'KEI=1 и Ma=0.3 являются диагностическими ориентирами. Сходимость моделей не устанавливает точность.'])
    return '\n'.join(lines)


def export_analysis(a, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    jp=unique_output(folder,'model_analysis','json');jp.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    cp=unique_output(folder,'sections','csv')
    with cp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(a['sections'][0]));writer.writeheader();writer.writerows(a['sections'])
    def table(headers, rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    hp=unique_output(folder,'model_analysis','html')
    body='<h1>Анализ расчётов продувки</h1><p>'+html.escape(a['created_moscow'])+' (Москва)</p>'
    body+=table(['Модель','P4, МПа абс.','Guo/AIR, %','KEI min','Ma max'],[[r['label'],round(r['P4_MPa'],6),round(r['delta_P4_vs_AIR_pct'],4),round(r['KEI_min'],4),round(r['Mach_max'],4)] for r in a['summary']])
    body+='<h2>Разбор</h2><pre>'+html.escape(analysis_text(a))+'</pre><h2>Полный паспорт расчёта</h2>'
    body+=table(['Параметр','Значение'],a['inputs'].items())
    body+='<h2>Источники и границы доказательств</h2><ul>'
    for item in a['sources']:body+='<li><a href="'+html.escape(item['url'],quote=True)+'">'+html.escape(item['source'])+'</a><p>'+html.escape(item['evidence'])+'</p></li>'
    body+='</ul><p>Подробные результаты всех каналов находятся в соседних JSON и CSV.</p>'
    hp.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Анализ продувки</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#163047}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:10px;border:1px solid #ccd9e2;text-align:left}th{background:#edf4f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}a{color:#135b88}</style>'+body+'</html>',encoding='utf-8')
    return [hp,jp,cp]


def analyze_flow_range(data, qmin, qmax, count):
    require_finite(qmin,qmax)
    if not 0<qmin<qmax or type(count) is not int or not 2<=count<=1001:raise ValueError('Неверный диапазон или число точек.')
    normalized(data);rows=[]
    for i in range(count):
        q=qmin+(qmax-qmin)*i/(count-1)
        try:
            a=analyze_models(data,q)
            for r in a['summary']:rows.append(dict(Q_m3_min=q,model=r['model'],status='computed',P4_MPa=r['P4_MPa'],delta_vs_AIR_pct=r['delta_P4_vs_AIR_pct'],KEI_min=r['KEI_min'],Mach_max=r['Mach_max']))
            for key,error in a['unavailable'].items():rows.append(dict(Q_m3_min=q,model=key,status='unavailable',reason=error))
        except (ValueError,ArithmeticError) as exc:rows.append(dict(Q_m3_min=q,model='all',status='unavailable',reason=str(exc)))
    summary={}
    for key in MODEL_LABELS:
        valid=[r for r in rows if r['model']==key and r['status']=='computed']
        summary[key]={'computed':len(valid),'requested':count}
        if valid:summary[key].update(P4_min_MPa=min(r['P4_MPa'] for r in valid),P4_max_MPa=max(r['P4_MPa'] for r in valid),
                                    KEI_below_one=sum(r['KEI_min']<1 for r in valid),Mach_ge_03=sum(r['Mach_max']>=.3 for r in valid))
    return dict(inputs=normalized(data),range=[qmin,qmax],count=count,summary=summary,rows=rows,
                note='Конечная сетка; экстремумы и прохождение критериев между узлами не доказаны. Число KEI<1 при ms=0 носит справочный характер.')


class Verification(Verification):
    def test_analysis_pressure_closure_and_decomposition(self):
        a=analyze_models(default_data())
        for key,r in a['raw_results'].items():
            total=sum(s['reverse_path_increment_MPa'] for s in a['sections'] if s['model']==key)
            self.assertAlmostEqual(total,r['P4']-default_data()['ATM_PRSSR_P0']/1e6,places=12)
        x=a['decomposition'];self.assertAlmostEqual(sum(x['parts'].values()),x['total_AIR_minus_Guo_shared_MPa'],places=14)
        self.assertIn('guo_nikuradse',a['unavailable'])
        self.assertEqual(set(a['raw_results']),{'air','guo_shared','guo_colebrook'})
    def test_colebrook_mode_and_joint_invariance(self):
        d=dict(default_data(),guo_friction_mode=2,pipe_roughness_m=.00004572,borehole_roughness_m=.0001,core_roughness_m=.0001)
        a=analyze_models(d)
        self.assertFalse(a['unavailable'])
        n=a['raw_results']['guo_nikuradse'];c=a['raw_results']['guo_colebrook']
        self.assertGreater(c['P4'],n['P4'])
        for s in c['sections']:self.assertAlmostEqual(s['friction'],s['friction_regime']['colebrook_f'],places=14)
        x,y=n['sections'][-1],c['sections'][-1]
        self.assertAlmostEqual(x['friction']*x['equivalent_length_m'],y['friction']*y['equivalent_length_m'],places=12)
        self.assertLess(abs(refinement_study(d)[-1]['relative_difference_percent']),.001)
    def test_colebrook_pressure_against_independent_integration(self):
        d=dict(default_data(),guo_friction_mode=2,pipe_roughness_m=.00004572,borehole_roughness_m=.0001,core_roughness_m=.0002)
        r=calculate(d,method='guo');mg=r['mass_gas_kg_s'];ms=r['mass_solids_kg_s'];RT=d['GAS_CONST_R']*d['well_temp_t']
        for s in r['sections']:
            up=s['flow']=='up';M=1+ms/mg if up else 1;f=s['friction']*(1+s['equivalent_length_m']/s['length_m'])
            def dp(p):
                rho=p/RT;v=mg/(rho*s['area_m2'])
                return M*rho*(9.81+(1 if up else -1)*f*v*v/(2*s['dh_m']))
            p=s['top_pressure_pa'] if up else s['bottom_pressure_pa'];dx=s['length_m']/2000*(1 if up else -1)
            for _ in range(2000):
                k1=dp(p);k2=dp(p+dx*k1/2);k3=dp(p+dx*k2/2);k4=dp(p+dx*k3);p+=dx*(k1+2*k2+2*k3+k4)/6
            target=s['bottom_pressure_pa'] if up else s['top_pressure_pa']
            self.assertLess(abs(p/target-1),1e-9)
    def test_flow_analysis_refuses_bad_ranges_and_records_missing_models(self):
        a=analyze_flow_range(default_data(),3,6,3)
        self.assertEqual(a['summary']['air']['computed'],3)
        self.assertEqual(a['summary']['guo_nikuradse']['computed'],0)
        with self.assertRaises(ValueError):analyze_flow_range(default_data(),6,3,3)
        a=analyze_models(dict(default_data(),source_literal_mode=1))
        self.assertIsNone(a['decomposition'])




# M3 phase 1: steady, vertical, constant-area reduction of the 2015 balances.
# This is NOT the author's transient Roe implementation or a verified full well.
from dataclasses import asdict

M3_SOURCE = 'https://scispace.com/pdf/pneumatic-transport-modeling-of-air-gas-drilling-hydraulics-1curcuqn82.pdf'
M3_DEFAULT_RTOL=2e-7

def m3_pressure_tolerance(rtol,pressure):
    require_finite(rtol,pressure)
    if not 0<rtol<=.01 or pressure<=0:raise ValueError('М3: неверный допуск или давление.')
    return max(1e-4,pressure*max(5e-12,.02*rtol))

M3_SCOPE = (
    'М3 v1.2: стационарная сухая вертикальная двухскоростная модель; '
    'раздельные скорости газа и частиц, ускорение обеих фаз. '
    'Участки и контур с заданными перепадами на стыках. '
    'Это стационарная адаптация уравнений Adeleke, не его нестационарный код.'
)


class M3DomainError(ValueError):
    """Physical/closure domain violation, not a computed pressure."""


@dataclass(frozen=True)
class M3Case:
    name: str
    length_m: float
    outer_diameter_m: float
    inner_diameter_m: float
    gas_mass_kg_s: float
    solids_mass_kg_s: float
    particle_diameter_m: float
    particle_density_kg_m3: float
    inlet_solids_velocity_m_s: float
    outlet_pressure_pa: float
    temperature_k: float
    viscosity_pa_s: float
    effective_roughness_m: float
    gas_R: float = 287.0
    gravity: float = 9.81
    solids_wall_normalization: str = 'yang_bulk'
    yang_regime: str = 'source_high_only'
    extra_fanning: float = 0.

    def __post_init__(self):
        try:
            require_finite(*(v for v in asdict(self).values() if isinstance(v, (int, float))))
        except (ValueError, TypeError) as exc:
            raise M3DomainError('М3: все числа должны быть конечными.') from exc
        if min(self.length_m, self.outer_diameter_m, self.gas_mass_kg_s,
               self.particle_diameter_m, self.particle_density_kg_m3,
               self.inlet_solids_velocity_m_s, self.outlet_pressure_pa,
               self.temperature_k, self.viscosity_pa_s, self.gas_R) <= 0:
            raise M3DomainError('М3: положительные параметры обязательны.')
        if not 0 <= self.inner_diameter_m < self.outer_diameter_m:
            raise M3DomainError('М3: неверные диаметры канала.')
        if min(self.solids_mass_kg_s, self.effective_roughness_m, self.gravity, self.extra_fanning) < 0:
            raise M3DomainError('М3: отрицательный расход, шероховатость или g.')
        if self.effective_roughness_m >= self.dh_m / 2:
            raise M3DomainError('М3: шероховатость слишком велика относительно Dh.')
        # Necessary geometric check for the spherical particles used by this model.
        if self.solids_mass_kg_s and self.particle_diameter_m >= self.minimum_width_m*(1-1e-12):
            raise M3DomainError('М3: сферическая частица не проходит свободно через канал.')
        if self.yang_regime not in ('source_high_only','yang_two_branch','adeleke_literal_high'):
            raise M3DomainError('М3: неизвестный режим корреляции Yang.')
        if self.solids_wall_normalization not in ('yang_bulk','legacy_intrinsic'):
            raise M3DomainError('М3: неизвестная нормировка трения частиц.')
        if self.solids_mass_kg_s:
            alpha = self.solids_mass_kg_s / (self.area_m2 * self.particle_density_kg_m3
                                           * self.inlet_solids_velocity_m_s)
            if not 0 < alpha <= 0.2:
                raise M3DomainError('М3: входная доля частиц вне принятого диапазона (0; 0.2].')

    @property
    def area_m2(self):
        return math.pi / 4 * ((self.outer_diameter_m-self.inner_diameter_m)*(self.outer_diameter_m+self.inner_diameter_m))

    @property
    def dh_m(self):
        return self.outer_diameter_m - self.inner_diameter_m

    @property
    def minimum_width_m(self):
        return self.dh_m / 2 if self.inner_diameter_m else self.dh_m


def m3_sphere_cd(re):
    """Eq. 3.21's continuous expression only. Do not guess its printed switch.

    The PDF prints a suspect 10**-5 switch. Use the actual continuous expression
    for Re <= 1e4, a project restriction, NOT an independently verified author range.
    The zero-slip force is evaluated separately without dividing by Re=0.
    """
    require_finite(re)
    if not 0 < re <= 1e4:
        raise M3DomainError('М3: Rep вне вычислительной ветви 0 < Rep <= 10000.')
    return 24/re + 4.152/re**0.343 + 0.413/(1 + 16300/re**1.09)


def m3_terminal_velocity(rho_g, rho_s, dp, mu, gravity=9.81):
    """Implicit solution of isolated-sphere force balance, Eq. 3.17.

    Numerical interpretation: Cd evaluated at rho_g * Ut * dp / mu, no wall factor.
    """
    if rho_s <= rho_g:
        raise M3DomainError('М3: плотность частицы должна превышать плотность газа.')
    if gravity == 0:
        return 0.0
    # Solve Cd(Re)*Re**2=C with a monotone safeguarded Newton iteration.
    # This replaces 65 repeated velocity bisections without changing the law.
    scale=rho_g*dp/mu
    target=4/3*(rho_s-rho_g)*gravity*dp/rho_g*scale**2
    def value_derivative(re):
        t=16300/re**1.09; d=1+t
        f=24*re+4.152*re**1.657+.413*re*re/d
        df=24+4.152*1.657*re**.657+.413*re*(2/d+1.09*t/d**2)
        return f,df
    lo=0.;hi=1e4
    if value_derivative(hi)[0] < target:
        raise M3DomainError('М3: Ut требует Rep выше выбранного диапазона.')
    re=min(hi,max(1e-15,min(target/24,math.sqrt(target/.44))))
    for _ in range(60):
        v,dv=value_derivative(re)
        if abs(v-target)<=2e-13*target:return re/scale
        if v>target:hi=re
        else:lo=re
        proposal=re-(v-target)/dv
        re=proposal if lo<proposal<hi else (lo+hi)/2
    raise M3DomainError('М3: не решён баланс терминальной скорости.')



def m3_drag(rho_g, eps, slip, dp, mu, solids_fraction=None):
    """Eqs. 3.18-3.19 with the explicitly printed Wen-Yu eps**-2.65 branch.

    Di Felice and the suspect positive exponent in Eq. 3.20 are NOT substituted.
    Returns signed force on solids per total mixture volume, N/m3.
    """
    if not 0 < eps <= 1:
        raise M3DomainError('М3: недопустимая пористость.')
    # alpha is conserved independently. Recovering it as 1-eps loses all
    # significant digits for trace loading, even while the particle ODE is finite.
    alpha=1-eps if solids_fraction is None else solids_fraction
    require_finite(rho_g,eps,slip,dp,mu,alpha)
    if min(rho_g,dp,mu)<=0 or not 0<=alpha<1:raise M3DomainError('М3: неверные данные межфазной силы.')
    if abs((eps+alpha)-1)>5e-15:raise M3DomainError('М3: несовместимые доли фаз.')
    if slip == 0 or alpha == 0:
        return 0.0, 0.0
    re = eps*rho_g*abs(slip)*dp/mu
    cd = m3_sphere_cd(re)
    force = 0.75*cd*eps*alpha*rho_g*abs(slip)*slip/dp * eps**-2.65
    return force, re


def m3_fanning(re, relative_roughness):
    """Eqs. 3.13-3.15. FANNING, not Darcy. Chen turbulent branch only.

    The first implementation deliberately excludes Re < 4000 for both pipe and
    annulus. This avoids treating pipe 16/Re as an exact annular laminar law.
    """
    require_finite(re,relative_roughness)
    if not 0<=relative_roughness<.5:raise M3DomainError('М3: недопустимая относительная шероховатость.')
    if re < 4000:
        raise M3DomainError('М3: Re < 4000; ламинарная/переходная ветвь не реализована.')
    rr = relative_roughness
    inner = rr**1.1098/2.8257 + 5.8506/re**0.898
    term = rr/3.7065 - 5.0452/re * math.log10(inner)
    return 1/(-4*math.log10(term))**2


def m3_local(case, pressure, solids_velocity):
    """State reconstructed from the TWO independently conserved mass fluxes.

    z points upwards. Wall and gravity terms oppose upward motion.
    yang_bulk: Fws = 2*f_F*alpha*rho_s*us**2/Dh. DOI10.1002/aic.690240326;
    DOE/MC/21023-2825 Vol.3 (1989), Appendix C pp.C-5,C-6,C-167:
    dFf=f_p*us**2/(2D)*dMs; dMs=alpha*rho_s*A*dz; f_p=4*f_F.
    DOE C-7 Eq16 explicitly includes alpha and states f_p=4*f_F.
    Adeleke Eq3.12 prints rho without an index. His actual code is unavailable.
    legacy_intrinsic reproduces our earlier unsupported normalization ONLY.
    """
    require_finite(pressure,solids_velocity)
    if pressure <= 0 or solids_velocity <= 0:
        raise M3DomainError('М3: неположительное давление или скорость шлама.')
    rho = pressure/(case.gas_R*case.temperature_k)
    alpha = (case.solids_mass_kg_s /
             (case.area_m2*case.particle_density_kg_m3*solids_velocity))
    eps = 1-alpha
    if not 0.8 <= eps <= 1:
        raise M3DomainError('М3: доля газа ниже 0.8; плотный режим не поддерживается.')
    if rho >= case.particle_density_kg_m3:
        raise M3DomainError('М3: плотность газа достигла плотности частиц.')
    vg = case.gas_mass_kg_s/(case.area_m2*rho*eps)
    slip = vg-solids_velocity
    re_g = eps*rho*vg*case.dh_m/case.viscosity_pa_s
    fg = m3_fanning(re_g, case.effective_roughness_m/case.dh_m)+case.extra_fanning
    wg = 2*fg*rho*vg**2/case.dh_m
    drag, re_p = m3_drag(rho, eps, slip, case.particle_diameter_m, case.viscosity_pa_s,alpha)
    fs, ws, ut, yang_branch = 0.0, 0.0, 0.0, 'no_solids'
    if case.solids_mass_kg_s:
        if slip <= 0 or case.gravity <= 0:
            raise M3DomainError('М3: ветвь Yang требует ug > us > 0 и g > 0.')
        ut = m3_terminal_velocity(rho, case.particle_density_kg_m3,
                                  case.particle_diameter_m, case.viscosity_pa_s, case.gravity)
        if case.solids_wall_normalization=='yang_bulk' and case.yang_regime=='source_high_only' and vg/ut <= 1.5:
            raise M3DomainError('М3: ветвь Yang 0.00315 требует ug/Ut > 1.5; другая ветвь не реализована.')
        yang_branch='high'
        coefficient,exponent=.00315,-.979
        if case.yang_regime=='yang_two_branch' and vg/ut<=1.5:
            # DOE/MC/21023-2825 Vol.III C-6: .0410, -1.021; fp=4*f_F.
            # Original Yang extension, absent from Adeleke Eq3.16. No smoothing.
            coefficient,exponent=.0410/4,-1.021;yang_branch='low'
        fs = coefficient*alpha/eps**3 * (alpha*ut/slip)**exponent
        wall_density = case.particle_density_kg_m3
        if case.solids_wall_normalization=='yang_bulk':wall_density *= alpha
        ws = 2*fs*wall_density*solids_velocity**2/case.dh_m
    gg = eps*rho*case.gravity
    gs = alpha*(case.particle_density_kg_m3-rho)*case.gravity
    modulus = 10**(-8.76*eps+5.43)
    sonic = 1-eps*rho*vg*vg/pressure
    if sonic <= 0.05:
        raise M3DomainError('М3: приближение к изотермической критической точке.')
    return dict(pressure_pa=pressure,gas_density_kg_m3=rho,voidage=eps,
                solids_volume_fraction=alpha,gas_velocity_m_s=vg,
                solids_velocity_m_s=solids_velocity,slip_m_s=slip,
                gas_Re=re_g,particle_Re=re_p,gas_fanning=fg,solids_fanning=fs,
                gas_wall_N_m3=wg,solids_wall_N_m3=ws,drag_N_m3=drag,
                gas_weight_N_m3=gg,solids_weight_N_m3=gs,
                solids_modulus_pa=modulus,isothermal_sonic_margin=sonic,
                terminal_velocity_m_s=ut,gas_to_terminal_ratio=vg/ut if ut else None,
                yang_branch=yang_branch,
                yang_extrapolated=bool(case.solids_mass_kg_s and case.yang_regime=='adeleke_literal_high' and vg/ut<=1.5))


def m3_derivative(case, p, us):
    """Exact steady reduction of the constant-area balances, NO fitted pressure.

    Js=ms/A; alpha=Js/(rho_s*us); eps'=alpha*us'/us;
    us'=(Fd-Fws-alpha*(rho_s-rho_g)*g)/(Js-G*alpha/us).
    p'=(-Fwg-eps*rho_g*g-Fd+rho_g*ug**2*eps')/
        (1-eps*rho_g*ug**2/p).
    Solids stress: dps/dz=-G(eps)*deps/dz. No energy/bed/variable-area model.
    """
    s = m3_local(case,p,us)
    dus, deps = 0.0, 0.0
    if case.solids_mass_kg_s:
        js = case.solids_mass_kg_s/case.area_m2
        den = js-s['solids_modulus_pa']*s['solids_volume_fraction']/us
        if den <= 1e-10*js:
            raise M3DomainError('М3: критическая точка стационарной твёрдой фазы.')
        dus = (s['drag_N_m3']-s['solids_wall_N_m3']-s['solids_weight_N_m3'])/den
        deps = s['solids_volume_fraction']*dus/us
    dp = (-s['gas_wall_N_m3']-s['gas_weight_N_m3']-s['drag_N_m3']
          +s['gas_density_kg_m3']*s['gas_velocity_m_s']**2*deps)/s['isothermal_sonic_margin']
    return dp,dus


def m3_rk4(case,y,h):
    def add(a,b,k):return (a[0]+k*b[0],a[1]+k*b[1])
    k1=m3_derivative(case,*y);k2=m3_derivative(case,*add(y,k1,h/2))
    k3=m3_derivative(case,*add(y,k2,h/2));k4=m3_derivative(case,*add(y,k3,h))
    return tuple(y[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(2))


def m3_integrate_rk4(case,inlet_pressure_pa,rtol=2e-5,early_outlet_event=False):
    """Adaptive RK4 step-doubling IVP, with domain rejection, no clipping states.

    early_outlet_event stops a shooting trial when p crosses prescribed outlet
    pressure DOWNWARDS before L. Its negative residual is a signed event distance,
    not an extrapolated downstream pressure. No failed domain trial is a root sign.
    """
    if not 0 < rtol <= 0.01:
        raise ValueError('М3: неверная точность.')
    if inlet_pressure_pa < case.outlet_pressure_pa:
        raise M3DomainError('М3: входное давление ниже заданного выходного.')
    x=0.0;y=(inlet_pressure_pa,case.inlet_solids_velocity_m_s)
    h=min(case.length_m/100,0.01);steps=0;rejected=0
    rows=[dict(z_m=x,**m3_local(case,*y))]
    while x < case.length_m:
        if steps+rejected > 60000:
            raise M3DomainError('М3: превышен лимит шагов, результат не принят.')
        h=min(h,case.length_m-x)
        try:
            full=m3_rk4(case,y,h)
            half=m3_rk4(case,y,h/2);fine=m3_rk4(case,half,h/2)
            m3_local(case,*fine)
            err=max(abs(fine[i]-full[i])/(15*(rtol*max(abs(y[i]),abs(fine[i]))+
                        (0.01 if i==0 else 1e-7))) for i in range(2))
        except (M3DomainError,OverflowError,ZeroDivisionError):
            rejected+=1;h/=2
            if h < 1e-9*min(1,case.length_m):
                raise M3DomainError('М3: граница области расчёта при z=%.9g м.'%x)
            continue
        if err > 1:
            h*=max(0.1,0.8*err**-0.2);rejected+=1;continue
        x+=h;y=fine;steps+=1
        row=dict(z_m=x,**m3_local(case,*y));rows.append(row)
        if early_outlet_event and y[0]<case.outlet_pressure_pa:
            return dict(residual=-case.outlet_pressure_pa*(case.length_m-x)/case.length_m
                        +(y[0]-case.outlet_pressure_pa),rows=rows,steps=steps,
                        rejected_steps=rejected,reached_outlet=False)
        h*=2 if err<0.01 else min(2,max(0.25,0.9*max(err,1e-12)**-0.2))
        h=min(h,case.length_m/20)
    return dict(residual=y[0]-case.outlet_pressure_pa,rows=rows,steps=steps,
                rejected_steps=rejected,reached_outlet=True)



def m3_shoot_root(evaluate,lower,upper,tolerance,upper_result=None):
    """Illinois bracketed secants with bisection safeguards, no extrapolation.

    A failed physical trial propagates as a failure, never as a residual sign.
    Acceptance requires BOTH a narrow pressure bracket and a reached endpoint
    with the prescribed outlet residual. Final profiles are recomputed by caller.
    """
    low=evaluate(lower);high=evaluate(upper) if upper_result is None else upper_result
    fa=low['residual'];fb=high['residual'];require_finite(lower,upper,tolerance,fa,fb)
    if not lower<upper or tolerance<=0 or fa>0 or fb<0:raise M3DomainError('М3: неверная скобка давления.')
    wa,wb=fa,fb;last=None;calls=1+(upper_result is None)
    for iteration in range(100):
        if upper-lower<=tolerance and high['reached_outlet'] and abs(fb)<=tolerance:
            return dict(pressure_pa=upper,result=high,iterations=iteration,evaluations=calls,bracket_width_pa=upper-lower)
        fraction=-wa/(wb-wa) if wb!=wa else .5
        # Illinois avoids stagnation. Periodic bisection bounds slow contraction.
        if not 1e-8<fraction<1-1e-8 or iteration%5==4:fraction=.5
        candidate=lower+(upper-lower)*fraction
        if candidate==lower or candidate==upper:break
        trial=evaluate(candidate);calls+=1;fc=trial['residual'];require_finite(fc)
        if fc>=0:
            upper=candidate;high=trial;fb=fc;wb=fb
            wa=wa*.5 if last=='upper' else fa;last='upper'
        else:
            lower=candidate;fa=fc;wa=fa
            wb=wb*.5 if last=='lower' else fb;last='lower'
    raise M3DomainError('М3: не достигнуты одновременно допуск давления и граничная невязка.')


def m3_solve(case,rtol=M3_DEFAULT_RTOL,inlet_condition='specified_velocity'):
    """Shoot inlet absolute pressure; inlet us is an explicit additional BC.

    Scope: straight section with specified roughness. Cross-section changes need
    extra matching conditions; sections are NOT automatically joined here.
    """
    if inlet_condition not in ('specified_velocity','developed_inlet'):
        raise ValueError('Неизвестное входное условие участка М3.')
    requested_case=case
    pressure_tol=m3_pressure_tolerance(rtol,case.outlet_pressure_pa)
    def trial_at(pressure,early):
        cc=case
        if inlet_condition=='developed_inlet' and cc.solids_mass_kg_s:
            cc=replace(cc,inlet_solids_velocity_m_s=m3_equilibrium_velocity(cc,pressure))
        return m3_integrate(cc,pressure,rtol,early)
    if inlet_condition=='specified_velocity':
        m3_local(case,case.outlet_pressure_pa,case.inlet_solids_velocity_m_s)
    lo=case.outlet_pressure_pa
    trial=trial_at(lo,True)
    if trial['residual']>0:
        raise M3DomainError('М3: нижняя граница давления не образует скобку.')
    hi=lo*1.15+1000;upper=None;valid_negative=lo
    for _ in range(24):
        try:
            upper=trial_at(hi,True)
        except M3DomainError as original_error:
            # A failed trial is NOT a pressure residual. Search the valid
            # portion of this pressure bracket before declaring a scope failure.
            invalid_upper=hi;upper=None
            for _probe in range(50):
                probe=(valid_negative+invalid_upper)/2
                try:
                    candidate=trial_at(probe,True)
                except M3DomainError:
                    invalid_upper=probe;continue
                if candidate['residual']>=0:
                    hi=probe;upper=candidate;break
                valid_negative=probe
            if upper is None:
                raise M3DomainError('М3: не найдена допустимая скобка до границы ветви; '+str(original_error)) from original_error
            break
        if upper['residual']>=0:break
        valid_negative=hi
        hi=lo+2*(hi-lo)
    else:raise M3DomainError('М3: давление не удалось заключить в скобку.')
    lo=valid_negative
    root=m3_shoot_root(lambda pressure:trial_at(pressure,True),lo,hi,pressure_tol,upper)
    hi=root['pressure_pa'];iteration=root['iterations']-1
    result=trial_at(hi,False)
    case=replace(case,inlet_solids_velocity_m_s=result['rows'][0]['solids_velocity_m_s'])
    if not result['reached_outlet'] or abs(result['residual'])>2*pressure_tol:
        raise M3DomainError('М3: не выполнена граничная точность давления.')
    result.update(inlet_pressure_pa=hi,pressure_drop_pa=hi-case.outlet_pressure_pa,
                  shooting_iterations=iteration+1,shooting_method='Illinois_with_bisection_safeguards',shooting_bracket_width_pa=root['bracket_width_pa'],numerical_tolerances=dict(rtol=rtol,pressure_boundary_pa=pressure_tol),case=asdict(case),requested_case=asdict(requested_case),inlet_condition=inlet_condition,scope=M3_SCOPE,
                  validation_status='numerically_checked_not_author_validated',
                  physical_validation=False,author_example_reproduced=False,
                  source_wall_normalization=case.solids_wall_normalization,
                  particle_to_width_ratio=case.particle_diameter_m/case.minimum_width_m,
                  notes=[
        'Входное условие частиц: '+inlet_condition+'; локальное равновесие является инженерным предположением.',
        'Wen-Yu выбран явно; Di Felice и спорный переключатель Cd не используются.',
        'Сила трения Yang нормирована на объём смеси через alpha*rho_s; авторский контроль Adeleke не завершён.',
        'Независимые кривые давления Narimatsu сравниваются в меню 21; выявлены расхождения, М3 физически не аттестована.',
        'Применимость к зазору 2 мм не подтверждена. Нормировка: '+case.solids_wall_normalization,
        'Эффективная шероховатость двух стенок задаётся пользователем.',
        'Доля газа >= 0.8 и Rep <= 10000 являются рабочими ограничениями, не авторскими гарантиями.',
        'Здесь показан отдельный участок. Весь контур доступен в пункте 20.'])
    mg,ms=case.gas_mass_kg_s,case.solids_mass_kg_s
    result['gas_mass_relative_error']=max(abs(r['voidage']*r['gas_density_kg_m3']*
        r['gas_velocity_m_s']*case.area_m2/mg-1) for r in result['rows'])
    result['diagnostics']=m3_profile_diagnostics(case,result['rows'])
    result['solids_mass_relative_error']=(max(abs(r['solids_volume_fraction']*
        case.particle_density_kg_m3*r['solids_velocity_m_s']*case.area_m2/ms-1)
        for r in result['rows']) if ms else 0.0)
    return result


def m3_demo_case():
    """Illustration using Temple geometry, NOT reproduction of Temple's experiment.

    Explicit project assumptions: smooth walls, inlet solids velocity=1 m/s,
    superficial inlet gas speed=8 m/s referenced at the outlet atmospheric density.
    """
    area=math.pi/4*(.0779**2-.0422**2)
    return M3Case('Иллюстрация на геометрии Temple; не авторский контроль',
        2.74,.0779,.0422,101325/(287*288.1)*8*area,.01514,
        .0003465,3150,1,101325,288.1,1.760e-5,0.0)


def _m3_base_published_control():
    """Source metadata, not simulated outputs masquerading as validation."""
    return dict(status='not_reproduced',source=M3_SOURCE,
        printed_pages='89-98; Tables 5.3-5.5, Figures 5.6-5.13',
        outer_m=.0779,inner_m=.0422,length_m=2.74,T_k=288.1,mu_pa_s=1.760e-5,
        particle_density_kg_m3=3150,particle_mean_m=.0003465,solids_kg_s=.01514,
        pending_to_check=['Установить входную скорость твёрдой фазы для данной серии',
                 'Уточнить шероховатость; нормировка Yang восстановлена отдельно от проверки кода Adeleke',
                 'Извлечь численные экспериментальные точки с погрешностью',
                 'Авторская нестационарная схема не реализована в стационарной адаптации'])


def m3_show(result):
    print('\n'+result['scope'])
    print('Вход: %.6f МПа абс.; выход: %.6f МПа абс.; dP=%.3f Па'%(
        result['inlet_pressure_pa']/1e6,result['case']['outlet_pressure_pa']/1e6,
        result['pressure_drop_pa']))
    print('Граничная невязка давления: %.6g Па'%result['residual'])
    print('z, м | p, МПа абс. | ug, м/с | us, м/с | доля шлама')
    rows=result['rows'];indices=sorted(set([0,len(rows)-1]+[i*(len(rows)-1)//8 for i in range(9)]))
    for i in indices:
        r=rows[i];print('%.5g | %.7f | %.5g | %.5g | %.6g'%(r['z_m'],r['pressure_pa']/1e6,
            r['gas_velocity_m_s'],r['solids_velocity_m_s'],r['solids_volume_fraction']))
    print('Отношение размера частицы к ширине прохода: %.6g'%result['particle_to_width_ratio'])
    print('Невязки массовых расходов газа/шлама: %.3g / %.3g'%(
        result['gas_mass_relative_error'],result['solids_mass_relative_error']))
    for item in result['notes']:print('* '+item)


def m3_input_case(data,flow):
    d,gas,q,mg,ms,_=project_common(data,flow)
    if abs(d['drilling_angle_alfa']-90)>1e-9:
        raise M3DomainError('М3 поддерживает только вертикальный канал.')
    if d['source_literal_mode']:
        raise M3DomainError('Для М3 выключите буквальный книжный режим AIR.')
    print('Паспорт: общий Q=%.6g м³/мин, mg=%.6g кг/с, ms=%.6g кг/с.'%(q,mg,ms))
    print('1. Вокруг бурильной трубы\n2. Вокруг колонковой трубы (узкий участок)')
    key=input('Участок: ').strip()
    if key=='1':
        inner=d['out_diam_dril_pipe_d'];L=d['dpth_h']-d['core_pipe_lngth_lk']
        pout=d['ATM_PRSSR_P0'];name='Затрубье бурильной трубы, отдельный участок'
    elif key=='2':
        inner=d['out_diam_core_pipe_dh'];L=d['core_pipe_lngth_lk']
        pout=number(input('Давление на ВЕРХНЕМ конце этого участка, Па абсолютных: '))
        name='Зазор вокруг колонковой трубы, экстраполяция без подтверждения'
    else:raise ValueError('Нет такого участка.')
    print('Скорость подачи частиц и шероховатость не заимствуются из коэффициентов AIR.')
    dp=number(input('Диаметр эквивалентной сферической частицы, мм: '))/1000
    us=number(input('Скорость шлама на НИЖНЕМ конце участка, м/с: '))
    rough=number(input('Эффективная шероховатость канала, мм (0 = гладкий канал): '))/1000
    mu=number(input('Динамическая вязкость воздуха, Па*с: '))
    return M3Case(name,L,d['brhl_diam_D'],inner,mg,ms,dp,d['rock_density_ro'],us,pout,
                  d['well_temp_t'],mu,rough,d['GAS_CONST_R'],d['ACCLRTN_GRAV_g'])


def m3_menu(data,flow):
    last=None
    while True:
        print('\n'+M3_SCOPE)
        print('1. Иллюстративный расчёт (все допущения показаны)\n2. Участок из общего паспорта\n'
              '3. Статус опубликованного контроля\n4. Экспорт последнего расчёта JSON и CSV\n0. Назад')
        try:
            c=input('Пункт М3: ').strip()
            if c=='0':return
            if c in ('1','2'):
                case=m3_demo_case() if c=='1' else m3_input_case(data,flow)
                print(json.dumps(asdict(case),ensure_ascii=False,indent=2))
                last=m3_solve(case);m3_show(last)
            elif c=='3':print(json.dumps(m3_published_control(),ensure_ascii=False,indent=2))
            elif c=='4':
                if last is None:raise ValueError('Сначала выполните расчёт.')
                folder=Path(__file__).resolve().parent
                jp=unique_output(folder,'m3_result','json')
                jp.write_text(json.dumps(last,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                cp=unique_output(folder,'m3_profile','csv')
                with cp.open('w',newline='',encoding='utf-8-sig') as f:
                    w=csv.DictWriter(f,fieldnames=list(last['rows'][0]));w.writeheader();w.writerows(last['rows'])
                print('Сохранено: '+str(jp)+'; '+str(cp))
        except (ValueError,ArithmeticError) as exc:print('Расчёт не принят: '+str(exc))
        except EOFError:return


class M3Verification(unittest.TestCase):
    def test_stokes_limit(self):
        self.assertLess(abs(m3_sphere_cd(1e-8)*1e-8/24-1),1e-5)

    def test_drag_exchange_conserves_total_momentum(self):
        for eps in (.8,.95,.999):
            for u in (-3.,0.,3.):
                force,_=m3_drag(1.2,eps,u,.00035,1.8e-5)
                self.assertGreaterEqual(force*u,0)
                self.assertAlmostEqual(force+m3_drag(1.2,eps,-u,.00035,1.8e-5)[0],0)

    def test_terminal_force_balance(self):
        u=m3_terminal_velocity(1.2,3150,.00035,1.8e-5)
        f=.75*m3_sphere_cd(1.2*u*.00035/1.8e-5)*1.2*u*u/.00035
        self.assertAlmostEqual(f,(3150-1.2)*9.81,places=7)

    def test_darcy_fanning_normalization(self):
        re=1e5;f=m3_fanning(re,0.001)
        # Independent Colebrook reference; Chen is an approximation.
        fd=friction_regime_check(290,.1,.001,.04,.00004)['colebrook_f']
        x=.03
        for _ in range(70):x=1/(-2*math.log10(.001/3.7+2.51/(re*math.sqrt(x))))**2
        self.assertLess(abs(4*f/x-1),.005)

    def test_steady_balances_independently_differentiated(self):
        c=m3_demo_case();p=110000.;us=1.0
        dp,du=m3_derivative(c,p,us);s=m3_local(c,p,us)
        h=1e-7
        a=m3_local(c,p-h*dp,us-h*du);b=m3_local(c,p+h*dp,us+h*du)
        gasflux=lambda r:r['voidage']*r['gas_density_kg_m3']*r['gas_velocity_m_s']**2+r['pressure_pa']
        dg=(gasflux(b)-gasflux(a))/(2*h)
        self.assertAlmostEqual(dg,-s['gas_wall_N_m3']-s['gas_weight_N_m3']-s['drag_N_m3'],delta=.01)
        js=c.solids_mass_kg_s/c.area_m2
        deps=s['solids_volume_fraction']*du/us
        ds=js*du-s['solids_modulus_pa']*deps
        self.assertAlmostEqual(ds,s['drag_N_m3']-s['solids_wall_N_m3']-s['solids_weight_N_m3'],places=8)

    def test_smooth_gas_only_exact_fanno_invariant(self):
        # g=0; constant mass flux and Chen f -> p^2-2*K*ln(p) has linear z.
        c=replace(m3_demo_case(),solids_mass_kg_s=0.,gravity=0.)
        p0=110000.;r=m3_integrate(c,p0,1e-7)
        f=m3_fanning(c.gas_mass_kg_s*c.dh_m/(c.area_m2*c.viscosity_pa_s),0.)
        K=(c.gas_mass_kg_s/c.area_m2)**2*c.gas_R*c.temperature_k
        invariant=lambda p:p*p-2*K*math.log(p)
        expected=invariant(p0)-4*f*K*c.length_m/c.dh_m
        self.assertLess(abs(invariant(r['rows'][-1]['pressure_pa'])/expected-1),1e-9)

    def test_gas_static_hydrostatic_limit(self):
        # No friction is not assumed: very low mass is intentionally excluded.
        with self.assertRaises(M3DomainError):
            m3_solve(replace(m3_demo_case(),gas_mass_kg_s=1e-7))

    def test_refinement_and_boundary(self):
        c=m3_demo_case();a=m3_solve(c,2e-5);b=m3_solve(c,2e-6)
        self.assertTrue(a['reached_outlet']);self.assertLess(abs(a['residual']),.1)
        self.assertLess(abs(a['inlet_pressure_pa']-b['inlet_pressure_pa']),.3)
        self.assertLess(a['gas_mass_relative_error'],1e-12)
        self.assertLess(a['solids_mass_relative_error'],1e-12)

    def test_forbidden_geometry_and_missing_validation(self):
        with self.assertRaises(M3DomainError):
            replace(m3_demo_case(),outer_diameter_m=.093,inner_diameter_m=.089,particle_diameter_m=.002)
        with self.assertRaises(M3DomainError):replace(m3_demo_case(),temperature_k=float('nan'))
        with self.assertRaises(M3DomainError):replace(m3_demo_case(),inlet_solids_velocity_m_s=.0001)
        self.assertEqual(m3_published_control()['status'],'not_reproduced')
        self.assertFalse(m3_solve(m3_demo_case())['physical_validation'])


class Verification(Verification,M3Verification):
    pass


# Numerical revision: L-stable two-stage SDIRK, gamma = 1 - 1/sqrt(2).
# Same m3_derivative; no change to force laws, coefficients or boundary conditions.
def m3_implicit_stage(case, base, initial, a):
    y = tuple(initial)
    scales = (max(1., abs(base[0])), max(.1, abs(base[1])))
    def residual(v):
        f = m3_derivative(case, *v)
        return tuple((v[i]-base[i]-a*f[i])/scales[i] for i in range(2))
    for _ in range(18):
        r = residual(y)
        norm = max(map(abs, r))
        if norm < 2e-11:
            return y
        columns = []
        for j in range(2):
            delta = 2e-6*max(abs(y[j]), scales[j]);column=None
            branch=m3_local(case,*y)['yang_branch'] if case.yang_regime=='yang_two_branch' else None
            for shrink in (1.,.1,.01):
                for direction in (1.,-1.):
                    increment=direction*delta*shrink
                    shifted=list(y);shifted[j]+=increment
                    try:
                        if branch is not None and m3_local(case,*shifted)['yang_branch']!=branch:continue
                        rr=residual(shifted)
                    except (ValueError,ArithmeticError):continue
                    column=tuple((rr[i]-r[i])/increment for i in range(2));break
                if column is not None:break
            if column is None:raise M3DomainError('М3: нет допустимой односторонней производной для шага Ньютона.')
            columns.append(column)
        A,B = columns[0][0],columns[1][0]
        C,D = columns[0][1],columns[1][1]
        det = A*D-B*C
        if not math.isfinite(det) or abs(det) < 1e-30:
            raise M3DomainError('М3: вырожденный шаг Ньютона.')
        step = ((D*r[0]-B*r[1])/det, (A*r[1]-C*r[0])/det)
        accepted = False
        for k in range(14):
            scale = 2.**-k
            trial = tuple(y[i]-scale*step[i] for i in range(2))
            try:
                newnorm = max(map(abs,residual(trial)))
            except (ValueError, ArithmeticError):
                continue
            if newnorm < norm:
                y = trial; accepted = True; break
        if not accepted:
            raise M3DomainError('М3: не сошёлся демпфированный шаг Ньютона.')
    raise M3DomainError('М3: превышено число итераций Ньютона.')


def m3_sdirk2(case, y, h):
    gamma = 1-1/math.sqrt(2)
    first = m3_implicit_stage(case,y,y,h*gamma)
    f1 = m3_derivative(case,*first)
    base = tuple(y[i]+h*(1-gamma)*f1[i] for i in range(2))
    return m3_implicit_stage(case,base,first,h*gamma)


# Preserve the earlier explicit integrator for independent small-case checks.


def m3_integrate(case, inlet_pressure_pa, rtol=M3_DEFAULT_RTOL, early_outlet_event=False):
    require_finite(inlet_pressure_pa,rtol)
    if not 0 < rtol <= .01:
        raise ValueError('М3: неверная точность.')
    if inlet_pressure_pa < case.outlet_pressure_pa:
        raise M3DomainError('М3: входное давление ниже заданного выходного.')
    if not case.solids_mass_kg_s:
        r=m3_integrate_rk4(case,inlet_pressure_pa,rtol,early_outlet_event)
        r.update(numerical_method='adaptive_RK4_pure_gas',rtol=rtol)
        return r
    x=0.; y=(inlet_pressure_pa,case.inlet_solids_velocity_m_s)
    h=min(.001,case.length_m/100); steps=0; rejected=0
    rows=[dict(z_m=x,**m3_local(case,*y))]
    while x < case.length_m:
        if steps+rejected > 20000:
            raise M3DomainError('М3: превышен лимит неявных шагов.')
        h=min(h,case.length_m-x)
        try:
            coarse=m3_sdirk2(case,y,h)
            half=m3_sdirk2(case,y,h/2); fine=m3_sdirk2(case,half,h/2)
            final_state=m3_local(case,*fine)
            err=max(abs(fine[i]-coarse[i])/(3*(rtol*max(abs(y[i]),abs(fine[i]))+
                        (.001 if i==0 else 1e-8))) for i in range(2))
        except (ValueError,ArithmeticError) as exc:
            rejected+=1; h/=2
            if h < 1e-10*min(1,case.length_m):
                raise M3DomainError('М3: граница при z=%.9g м: '%x+str(exc)) from exc
            continue
        if err > 1:
            h*=max(.1,.8*err**(-1/3));rejected+=1;continue
        x+=h; y=fine; steps+=1
        row=dict(z_m=x,**final_state)
        if early_outlet_event and len(rows)>1:rows[-1]=row
        else:rows.append(row)
        if early_outlet_event and y[0] < case.outlet_pressure_pa:
            return dict(residual=-case.outlet_pressure_pa*(case.length_m-x)/case.length_m+
                        y[0]-case.outlet_pressure_pa, rows=rows,steps=steps,
                        rejected_steps=rejected,reached_outlet=False,numerical_method='SDIRK2')
        h*=min(2.,max(.25,.85*max(err,1e-12)**(-1/3)))
        h=min(h,case.length_m/40)
    return dict(residual=y[0]-case.outlet_pressure_pa,rows=rows,steps=steps,
                rejected_steps=rejected,reached_outlet=True,numerical_method='SDIRK2',rtol=rtol)


def comparison_scenario_defaults():
    """Research assumptions, not measured additions to Kudryashov's passport."""
    return dict(pipe_roughness_m=4.572e-5,borehole_roughness_m=1e-4,
                core_roughness_m=1e-4,particle_diameter_m=.0003465,
                inlet_solids_velocity_m_s=1.)


def compare_three_models(data, flow=None, additions=None, rtol=M3_DEFAULT_RTOL):
    d,gas,q,mg,ms,_=project_common(data,flow)
    if d['source_literal_mode'] or d['up_formula_variant']!=1 or abs(d['drilling_angle_alfa']-90)>1e-9:
        raise ValueError('Сравнение М1/М2/М3 требует общих условий, варианта AIR=1 и вертикальности.')
    extras=comparison_scenario_defaults() if additions is None else dict(additions)
    required=set(comparison_scenario_defaults())
    if set(extras)!=required:raise ValueError('Нужны все пять дополнительных параметров сценария.')
    require_finite(*extras.values())
    d.update({k:extras[k] for k in ('pipe_roughness_m','borehole_roughness_m','core_roughness_m')})
    full={}
    for key,method,mode in [('M1','air',0),('M2_shared','guo',0),('M2_Nikuradse','guo',1),('M2_Colebrook','guo',2)]:
        full[key]=calculate(dict(d,guo_friction_mode=mode),q,method)
    # Narrow-section common pressure is deliberately the M2-shared P1 prediction.
    # This is a numerical boundary condition, not an observation or M3 interface.
    narrow_top=full['M2_shared']['P1']*1e6
    viscosity=friction_regime_check(d['well_temp_t'],mg,1.,1.,None)['viscosity_Pa_s']
    blocks=[]
    for name,L,inner,pout in [
        ('wide',d['dpth_h']-d['core_pipe_lngth_lk'],d['out_diam_dril_pipe_d'],d['ATM_PRSSR_P0']),
        ('narrow',d['core_pipe_lngth_lk'],d['out_diam_core_pipe_dh'],narrow_top)]:
        area,dh=annulus(d['brhl_diam_D'],inner)
        rough=(extras['pipe_roughness_m']*inner+extras['borehole_roughness_m']*d['brhl_diam_D'])/(inner+d['brhl_diam_D'])
        f=.009407/dh**(1/3)
        b=gas.g/(2*gas.R*d['well_temp_t']);load=ms/mg
        pressure=pout
        for _ in range(int(d['air_substeps'])):
            length=L/int(d['air_substeps'])
            pressure=(math.sqrt(pressure**2+f*(1+d['gast_koef_kr']*load)*mg**2*gas.R*d['well_temp_t']*length/(dh*area**2))+
                      pressure*(1+load)*b*length)/(1-(1+load)*b*length)
        rows=[]
        def record(key,pin,us_top=None,us_bottom=None,alpha_top=None):
            vg_top=mg*gas.R*d['well_temp_t']/(area*pout)
            vg_bottom=mg*gas.R*d['well_temp_t']/(area*pin)
            rows.append(dict(model=key,p_top_pa=pout,p_bottom_pa=pin,delta_p_pa=pin-pout,
                             gas_superficial_top_m_s=vg_top,gas_superficial_bottom_m_s=vg_bottom,
                             solids_top_m_s=us_top,solids_bottom_m_s=us_bottom,solids_fraction_top=alpha_top))
        record('M1',pressure)
        section=Section(name,L,area,dh,f,d['well_temp_t'],d['well_temp_t'])
        for key,ff in [('M2_shared',f),('M2_Nikuradse',nikuradse(rough,dh)),
                       ('M2_Colebrook',friction_regime_check(d['well_temp_t'],mg,area,dh,rough)['colebrook_f'])]:
            record(key,section_pressure(pout,replace(section,friction_darcy=ff),gas,mg,ms))
        case=M3Case(name,L,d['brhl_diam_D'],inner,mg,ms,extras['particle_diameter_m'],d['rock_density_ro'],
                    extras['inlet_solids_velocity_m_s'],pout,d['well_temp_t'],viscosity,rough,gas.R,gas.g)
        try:
            m3=m3_solve(case,rtol)
            last=m3['rows'][-1]
            record('M3',m3['inlet_pressure_pa'],last['solids_velocity_m_s'],case.inlet_solids_velocity_m_s,last['solids_volume_fraction'])
            rows[-1].update(gas_intrinsic_top_m_s=last['gas_velocity_m_s'],physical_validation=False)
        except (ValueError,ArithmeticError) as exc:
            m3=dict(error=str(exc),case=asdict(case),physical_validation=False)
            rows.append(dict(model='M3',error=str(exc),p_top_pa=pout,p_bottom_pa=None,delta_p_pa=None))
        blocks.append(dict(section=name,area_m2=area,dh_m=dh,length_m=L,
                           effective_roughness_m=rough,common_top_pressure_pa=pout,
                           narrow_boundary_origin='M2_shared_full_P1' if name=='narrow' else 'passport_outlet',
                           rows=rows,m3=m3))
    return dict(created_moscow=timestamp(),passport=d,additions=extras,Q_reference_m3_min=q,
                mass_gas_kg_s=mg,mass_solids_kg_s=ms,viscosity_pa_s=viscosity,
                full_models=full,sections=blocks,M3_full_circuit_available=False,
                status='scenario_comparison_not_physical_validation',
                notes=['М2_shared сохраняет трение AIR; М2_Nikuradse использует 6.19; Colebrook является модификацией.',
                       'Добавочные данные заданы как сценарий и не являются измерениями нашей скважины.',
                       'Два участка М3 независимы. Переход между ними, коронка и полный контур не рассчитаны.',
                       'Нельзя складывать перепады независимых участков М3 в полный результат.'])


def comparison_text(result):
    lines=['СЦЕНАРНОЕ СРАВНЕНИЕ М1/М2/М3',
           'Общий Q=%.8f м3/мин; mg=%.9f кг/с; ms=%.9f кг/с.'%(result['Q_reference_m3_min'],result['mass_gas_kg_s'],result['mass_solids_kg_s']),
           'Все давления абсолютные. Добавления: '+json.dumps(result['additions'],ensure_ascii=False)]
    for block in result['sections']:
        lines.append('\nУчасток %s, L=%g м, p_top=%.6f МПа'%(block['section'],block['length_m'],block['common_top_pressure_pa']/1e6))
        for row in block['rows']:
            if row.get('error'):lines.append(row['model']+': '+row['error'])
            else:lines.append('%-16s p_bottom=%.6f МПа, dP=%.3f кПа'%(row['model'],row['p_bottom_pa']/1e6,row['delta_p_pa']/1e3))
    lines.append('\nПолный контур:')
    for key,r in result['full_models'].items():lines.append('%-16s P4=%.6f МПа, Pres=%.6f МПа'%(key,r['P4'],r['Pres']))
    lines.append('М3: полный контур пока недоступен.')
    lines.extend(result['notes'])
    return '\n'.join(lines)


def three_model_menu(data,flow):
    extras=comparison_scenario_defaults()
    print('Дополнения ниже являются СЦЕНАРНЫМИ предположениями. Они не измерены.')
    print(json.dumps(extras,ensure_ascii=False,indent=2))
    raw=input('Enter = этот сценарий; e = изменить все дополнения: ').strip().lower()
    if raw=='e':
        for key in extras:extras[key]=number(input(key+' (СИ): '))
    elif raw:raise ValueError('Ожидается Enter или e.')
    print('Расчёт двух независимых участков. Полный контур М3 не заявляется.')
    result=compare_three_models(data,flow,extras)
    print(comparison_text(result))
    path=unique_output(Path(__file__).resolve().parent,'three_model_comparison','json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Сохранено: '+str(path))


class Verification(Verification):
    def test_implicit_matches_previous_explicit_short_case(self):
        case=m3_demo_case()
        a=m3_integrate(case,110000.,2e-7)
        b=m3_integrate_rk4(case,110000.,2e-7)
        self.assertLess(abs(a['rows'][-1]['pressure_pa']-b['rows'][-1]['pressure_pa']),.1)
        self.assertLess(abs(a['rows'][-1]['solids_velocity_m_s']-b['rows'][-1]['solids_velocity_m_s']),1e-5)



# Source-backed wall normalization audit. All operational calculations remain here.
M3_WALL_SOURCE = 'https://www.osti.gov/servlets/purl/6955584'

def m3_source_audit(data, flow=None):
    base=compare_three_models(data,flow,rtol=2e-6)
    rows=[]
    for b in base['sections']:
        case=M3Case(**b['m3']['case'])
        current=b['m3']
        old=m3_solve(replace(case,solids_wall_normalization='legacy_intrinsic'),2e-6)
        ratio=min(v['gas_velocity_m_s']/m3_terminal_velocity(v['gas_density_kg_m3'],case.particle_density_kg_m3,case.particle_diameter_m,case.viscosity_pa_s,case.gravity) for v in current['rows'])
        rows.append(dict(section=b['section'],legacy_p_bottom_pa=old['inlet_pressure_pa'],
            yang_bulk_p_bottom_pa=current['inlet_pressure_pa'],legacy_drop_pa=old['pressure_drop_pa'],
            yang_bulk_drop_pa=current['pressure_drop_pa'],minimum_ug_over_Ut=ratio,
            M2_same_gas_Chen_p_bottom_pa=section_pressure(case.outlet_pressure_pa, Section('Guo_with_same_Chen_diagnostic',case.length_m,case.area_m2,case.dh_m,4*current['rows'][0]['gas_fanning'],case.temperature_k,case.temperature_k),Gas(R=case.gas_R,g=case.gravity),case.gas_mass_kg_s,case.solids_mass_kg_s),
            solids_top_m_s=current['rows'][-1]['solids_velocity_m_s'],
            concentration_top=current['rows'][-1]['solids_volume_fraction']))
    temple=[]
    for mode in ('legacy_intrinsic','yang_bulk'):
        c=replace(m3_demo_case(),solids_wall_normalization=mode)
        r=m3_solve(c,2e-6)
        temple.append(dict(normalization=mode,pressure_drop_pa=r['pressure_drop_pa'],
                           solids_outlet_m_s=r['rows'][-1]['solids_velocity_m_s'],
                           author_example_reproduced=False))
    return dict(created_moscow=timestamp(),source=M3_WALL_SOURCE,
        source_locations='Appendix C: C-5 Eq6, C-6 Eq8/11, C-7 Eq16, C-167 nomenclature',
        formula='Fws=2*f_F*alpha*rho_s*us^2/Dh; f_p=4*f_F',
        passport_comparison=base,normalization_comparison=rows,temple_illustration=temple,
        status='wall_normalization_repaired_author_example_still_not_reproduced',
        note='Наше исправление нормировки Yang не устанавливает фактическую реализацию кода Adeleke.')

def m3_source_audit_menu(data,flow):
    print('Аудит Yang: исправленная нормировка и историческая ошибка нашей реконструкции.')
    print('Коэффициенты и входные данные по результатам не подбираются. Расчёт может занять несколько минут.')
    audit=m3_source_audit(data,flow)
    for row in audit['normalization_comparison']:print(json.dumps(row,ensure_ascii=False))
    print('Иллюстрация Temple, не авторская верификация:',audit['temple_illustration'])
    p=unique_output(Path(__file__).resolve().parent,'m3_source_audit','json')
    p.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Сохранено:',str(p))

class Verification(Verification):
    def test_valid_solution_below_invalid_first_bracket_trial(self):
        # At the initial high trial this case leaves ug/Ut>1.5, while the root
        # itself belongs to the supported branch. Invalid probes are not signs.
        c=replace(m3_demo_case(),particle_diameter_m=.0007135,
                  gas_mass_kg_s=m3_demo_case().gas_mass_kg_s*9.5/8)
        with self.assertRaises(M3DomainError):
            m3_local(c,c.outlet_pressure_pa*1.15+1000,c.inlet_solids_velocity_m_s)
        r=m3_solve(c,2e-6)
        self.assertTrue(r['reached_outlet']);self.assertLess(abs(r['residual']),.1)

    def test_yang_force_matches_independent_original_mass_form(self):
        c=m3_demo_case();p=110000.;us=1.
        for factor in (.1,1.,3.):
            cc=replace(c,solids_mass_kg_s=c.solids_mass_kg_s*factor)
            state=m3_local(cc,p,us);eps=state['voidage'];alpha=1-eps
            ut=m3_terminal_velocity(state['gas_density_kg_m3'],cc.particle_density_kg_m3,
                                    cc.particle_diameter_m,cc.viscosity_pa_s)
            fp=.0126*alpha/eps**3*(alpha*ut/(state['gas_velocity_m_s']-us))**(-.979)
            # DOE C-5 Eq6, divide force on A*dz by A*dz.
            expected=fp*us**2/(2*cc.dh_m)*(cc.solids_mass_kg_s/(cc.area_m2*us))
            self.assertAlmostEqual(state['solids_wall_N_m3'],expected,delta=abs(expected)*1e-10)

    def test_legacy_force_ratio_exposes_missing_fraction(self):
        c=m3_demo_case();a=m3_local(c,110000.,1.)
        b=m3_local(replace(c,solids_wall_normalization='legacy_intrinsic'),110000.,1.)
        self.assertAlmostEqual(a['solids_wall_N_m3']/b['solids_wall_N_m3'],a['solids_volume_fraction'],places=12)

    def test_yang_regime_and_normalization_guards(self):
        with self.assertRaises(M3DomainError):replace(m3_demo_case(),solids_wall_normalization='unknown')
        c=replace(m3_demo_case(),inlet_solids_velocity_m_s=.1)
        # Keep turbulent gas Re, lower ug by increasing density.
        with self.assertRaises(M3DomainError):m3_local(c,5e6,.1)


# M3 v1.0 completion: coupled sections, explicit interface data and diagnostics.
# Public interface contracts use SI and ABSOLUTE pressure throughout.
M3_VERSION='1.2-numerical-quality'
M3_SOURCE_RECORDS=[
 dict(name='Adeleke, 2015, PhD thesis',url='https://etda.libraries.psu.edu/catalog/26245',
      evidence='Chapter 3 balances and closures; Chapter 5 Temple geometry and FCC minima. Stationary reduction implemented here; no claim of Roe/WENO reproduction.'),
 dict(name='Robertson et al., DOE/MC/21023-2825 Vol.III, 1989',url='https://www.osti.gov/servlets/purl/6955584',
      evidence='Appendix C: C-5 Eq6, C-6 Eqs10-11, C-7 Eq16 and fp=4fs; C-167 velocities. Low branch is an optional Yang extension, not Adeleke Eq3.16.'),
 dict(name='Turton & Levenspiel, 1986',url='https://doi.org/10.1016/0032-5910(86)80012-2',
      evidence='Continuous sphere-drag correlation. The suspect 10**-5 switch printed in Adeleke is not adopted.'),
 dict(name='Haider & Levenspiel, 1989',url='https://doi.org/10.1016/0032-5910(89)80008-7',
      evidence='Eq3 reproduces the continuous Turton-Levenspiel formula and Table1 gives its Reynolds range; this independently resolves the suspect printed 10**-5 switch.'),
 dict(name='Temple et al., 1996',url='https://doi.org/10.2118/37327-MS',
      evidence='Experimental pneumatic transport in a vertical annulus. Numerical minima below are transcribed from Adeleke Table5.6, not a fresh reading of the SPE raw data.')]


def m3_profile_diagnostics(case,rows):
    """Sampled profile checks; no claim to find extrema between accepted nodes."""
    if not rows:raise ValueError('М3: пустой профиль.')
    result={}
    for field in ('pressure_pa','gas_velocity_m_s','solids_velocity_m_s',
                  'solids_volume_fraction','gas_Re','particle_Re','isothermal_sonic_margin'):
        result[field+'_min']=min(r[field] for r in rows)
        result[field+'_max']=max(r[field] for r in rows)
    ratios=[r['gas_to_terminal_ratio'] for r in rows if r['gas_to_terminal_ratio'] is not None]
    result['gas_to_terminal_ratio_min']=min(ratios) if ratios else None
    result['yang_branches']=sorted(set(r['yang_branch'] for r in rows))
    result['yang_extrapolated_nodes']=sum(bool(r.get('yang_extrapolated',False)) for r in rows)
    result['maximum_gas_Mach_gamma_1_4']=max(r['gas_velocity_m_s']/math.sqrt(1.4*case.gas_R*case.temperature_k) for r in rows)
    # A posteriori independent balance quadrature, with an explicit quadrature error.
    # Using trapezoids makes this diagnostic independent of SDIRK stage weights.
    fields=('gas_wall_N_m3','gas_weight_N_m3','drag_N_m3','solids_wall_N_m3','solids_weight_N_m3')
    integrals={key:sum((b['z_m']-a['z_m'])*(a[key]+b[key])/2 for a,b in zip(rows,rows[1:])) for key in fields}
    first,last=rows[0],rows[-1];jg=case.gas_mass_kg_s/case.area_m2;js=case.solids_mass_kg_s/case.area_m2
    gas_momentum_change=last['pressure_pa']-first['pressure_pa']+jg*(last['gas_velocity_m_s']-first['gas_velocity_m_s'])
    solid_momentum_change=js*(last['solids_velocity_m_s']-first['solids_velocity_m_s'])+(last['solids_modulus_pa']-first['solids_modulus_pa'])/(8.76*math.log(10))
    result['force_integrals_pa_trapezoidal']=integrals
    result['gas_momentum_balance_quadrature_residual_pa']=gas_momentum_change+integrals['gas_wall_N_m3']+integrals['gas_weight_N_m3']+integrals['drag_N_m3']
    result['solids_momentum_balance_quadrature_residual_pa']=solid_momentum_change-integrals['drag_N_m3']+integrals['solids_wall_N_m3']+integrals['solids_weight_N_m3']
    result['quadrature_note']='Остаток включает погрешность трапеций на сохранённой адаптивной сетке; это не давление подгонки.'
    return result


def m3_equilibrium_velocity(case,pressure):
    """Developed-inlet alternative: Fdrag=Fwall+Fweight at the specified p.

    This is an EXPLICIT engineering boundary assumption, not an author datum.
    No interpolation across invalid states or across the discontinuous Yang switch.
    """
    if not case.solids_mass_kg_s:return case.inlet_solids_velocity_m_s
    js=case.solids_mass_kg_s/case.area_m2
    lo=max(js/(case.particle_density_kg_m3*.2)*(1+1e-9),.001)
    rho=pressure/(case.gas_R*case.temperature_k)
    hi=case.gas_mass_kg_s/(case.area_m2*rho)+js/case.particle_density_kg_m3
    previous=None;brackets=[]
    for i in range(121):
        us=lo+(hi-lo)*i/120
        try:
            st=m3_local(case,pressure,us)
            value=st['drag_N_m3']-st['solids_wall_N_m3']-st['solids_weight_N_m3']
        except (ValueError,ArithmeticError):previous=None;continue
        point=(us,value,st['yang_branch'])
        if previous and previous[2]==point[2] and previous[1]*value<=0:brackets.append((previous,point))
        previous=point
    if len(brackets)!=1:raise M3DomainError('М3: равновесная входная скорость не имеет единственного допустимого корня.')
    left,right=brackets[0]
    for _ in range(55):
        us=(left[0]+right[0])/2;st=m3_local(case,pressure,us)
        value=st['drag_N_m3']-st['solids_wall_N_m3']-st['solids_weight_N_m3']
        if st['yang_branch']!=left[2]:raise M3DomainError('М3: равновесный корень пересёк разрыв ветвей Yang.')
        if left[1]*value<=0:right=(us,value,st['yang_branch'])
        else:left=(us,value,st['yang_branch'])
    return (left[0]+right[0])/2


@dataclass(frozen=True)
class M3Passport:
    """Additional inputs. No unmeasured loss is silently treated as known zero.

    Recovery sign: p_after_wide-p_before_narrow at the upward expansion.
    Each supplied pressure jump is placed at its actual interface. Connector K is
    referred to the drill-pipe velocity and spread uniformly along its length.
    Zero interface jumps form an explicit idealized scenario, not a bit prediction.
    """
    particle_diameter_m: float
    maximum_particle_diameter_m: float
    inlet_solids_velocity_m_s: float
    viscosity_pa_s: float
    steel_roughness_m: float
    borehole_roughness_m: float
    core_roughness_m: float
    return_transition_pressure_recovery_pa: float
    bit_pressure_drop_pa: float
    core_interface_pressure_addition_pa: float
    barrel_pipe_interface_pressure_addition_pa: float
    connection_total_K: float
    inlet_condition: str='specified_velocity'
    yang_regime: str='source_high_only'
    eccentricity_m: float=0.
    data_status: str='user_inputs_not_experimentally_validated'

    def __post_init__(self):
        values=asdict(self)
        for key,value in values.items():
            if key in ('inlet_condition','yang_regime','data_status'):continue
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('М3: неверное числовое поле '+key)
        if min(self.particle_diameter_m,self.maximum_particle_diameter_m,self.inlet_solids_velocity_m_s,self.viscosity_pa_s)<=0:raise ValueError('М3: размер, скорость и вязкость должны быть положительными.')
        if self.maximum_particle_diameter_m<self.particle_diameter_m:raise ValueError('Максимальная частица меньше средней.')
        if min(self.steel_roughness_m,self.borehole_roughness_m,self.core_roughness_m,self.eccentricity_m,self.bit_pressure_drop_pa,self.connection_total_K)<0:raise ValueError('Отрицательная шероховатость или эксцентриситет.')
        if self.inlet_condition not in ('specified_velocity','developed_inlet'):raise ValueError('Неизвестное входное условие М3.')
        if self.yang_regime not in ('source_high_only','yang_two_branch','adeleke_literal_high'):raise ValueError('Неизвестный режим Yang.')
        if self.eccentricity_m:raise M3DomainError('М3: эксцентричное затрубье пока не поддерживается; расчёт концентрического канала запрещён для этих данных.')
        if not isinstance(self.data_status,str) or not self.data_status.strip():raise ValueError('Укажите статус дополнительных данных.')


def m3_example_passport(data):
    d=normalized(data)
    _,geom=project_channels(d)
    return M3Passport(particle_diameter_m=.0003465,maximum_particle_diameter_m=.0003465,
        inlet_solids_velocity_m_s=1.,viscosity_pa_s=friction_regime_check(d['well_temp_t'],.1,1.,1.,None)['viscosity_Pa_s'],
        steel_roughness_m=.00004572,borehole_roughness_m=.0001,core_roughness_m=.0001,
        return_transition_pressure_recovery_pa=0.,bit_pressure_drop_pa=0.,
        core_interface_pressure_addition_pa=0.,barrel_pipe_interface_pressure_addition_pa=0.,
        connection_total_K=geom['connection_count']*geom['xi_each'],
        data_status='ILLUSTRATION: particle size, roughness and inlet speed assumed; connector K from common AIR geometry rule; other interface static jumps explicitly zero')



def m3_gas_supply_section(bottom_pressure,case,rtol=1e-8):
    """Known bottom p, downward pure gas, integrate upwards along z.

    dp/dz=(-rho*g+2*fF*rho*u**2/Dh)/(1-rho*u**2/p).
    Acceleration retained; Chen only, no particles, no use of M1/M2 result.
    """
    if case.solids_mass_kg_s:raise ValueError('В нисходящем канале М3 допускается только чистый газ.')
    if not 0<rtol<.01 or not math.isfinite(bottom_pressure) or bottom_pressure<=0:raise ValueError('Неверное давление или допуск.')
    def rhs(p):
        st=m3_local(case,p,case.inlet_solids_velocity_m_s)
        return (-st['gas_weight_N_m3']+st['gas_wall_N_m3'])/st['isothermal_sonic_margin']
    def step(p,h):
        a=rhs(p);b=rhs(p+h*a/2);c=rhs(p+h*b/2);d=rhs(p+h*c)
        return p+h*(a+2*b+2*c+d)/6
    z=0.;p=bottom_pressure;h=min(.1,case.length_m/20);n=0;rejected=0
    rows=[dict(z_m=0.,**m3_local(case,p,case.inlet_solids_velocity_m_s))]
    while z<case.length_m:
        if n+rejected>20000:raise M3DomainError('М3: лимит шагов нисходящего газа.')
        h=min(h,case.length_m-z)
        try:
            full=step(p,h);half=step(p,h/2);fine=step(half,h/2)
            m3_local(case,fine,case.inlet_solids_velocity_m_s)
            err=abs(fine-full)/(15*(.00001+rtol*max(p,fine)))
        except (ValueError,ArithmeticError) as exc:
            h/=2;rejected+=1
            if h<1e-10:raise M3DomainError('М3: недопустим нисходящий канал: '+str(exc)) from exc
            continue
        if err>1:h*=max(.1,.8*err**-.2);rejected+=1;continue
        p=fine;z+=h;n+=1
        rows.append(dict(z_m=z,**m3_local(case,p,case.inlet_solids_velocity_m_s)))
        h*=min(2.,max(.25,.9*max(err,1e-12)**-.2));h=min(h,case.length_m/40)
    return dict(name=case.name,case=asdict(case),bottom_pressure_pa=bottom_pressure,top_pressure_pa=p,rows=rows,
                steps=n,rejected_steps=rejected,direction='gas_down_profile_coordinate_up')


def m3_full_circuit(data,flow,extra,rtol=M3_DEFAULT_RTOL):
    """Full conditional hydraulic circuit on shared mass flows and geometry.

    A single shooting pressure solves BOTH return sections. Particle velocity
    is carried through the interface, never reset. The unresolved sharp-interface
    mechanics is supplied as an explicit static pressure jump, not invented K.
    """
    if isinstance(extra,dict):extra=M3Passport(**extra)
    if not isinstance(extra,M3Passport):raise ValueError('Нужен дополнительный паспорт М3.')
    if extra.yang_regime=='adeleke_literal_high':
        raise M3DomainError('Неограниченная верхняя ветвь Yang разрешена только для явно отмеченного исследования отдельного участка; не для итогового контура.')
    d,gas,q,mg,ms,_=project_common(data,flow)
    pressure_tol=m3_pressure_tolerance(rtol,d['ATM_PRSSR_P0'])
    if d['source_literal_mode'] or d['geometry_mode']!=1 or abs(d['drilling_angle_alfa']-90)>1e-9:
        raise M3DomainError('М3: нужны общий режим, полная геометрия и вертикальный ствол.')
    Ln=d['core_pipe_lngth_lk'];Lw=d['dpth_h']-Ln
    if min(Ln,Lw)<=0:raise M3DomainError('М3: длины обоих участков затрубья должны быть положительными.')
    if not d['core_out_bk'] and extra.core_interface_pressure_addition_pa:
        raise ValueError('При отсутствии керна отдельный скачок на входе канала вокруг керна должен быть нулевым.')
    gap=(d['brhl_diam_D']-d['out_diam_core_pipe_dh'])/2
    if ms and extra.maximum_particle_diameter_m>=gap*(1-1e-12):raise M3DomainError('М3: максимальная частица не проходит зазор. Среднего диаметра недостаточно для проверки проходимости.')
    def rough(outer,inner,ro,ri):return (outer*ro+inner*ri)/(outer+inner) if inner else ro
    def make(name,L,outer,inner,solids=ms,roughness=None):
        return M3Case(name,L,outer,inner,mg,solids,extra.particle_diameter_m,d['rock_density_ro'],
            extra.inlet_solids_velocity_m_s,d['ATM_PRSSR_P0'],d['well_temp_t'],extra.viscosity_pa_s,
            extra.steel_roughness_m if roughness is None else roughness,gas.R,gas.g,'yang_bulk',extra.yang_regime)
    narrow=make('Возврат: вокруг колонковой трубы',Ln,d['brhl_diam_D'],d['out_diam_core_pipe_dh'],roughness=rough(d['brhl_diam_D'],d['out_diam_core_pipe_dh'],extra.borehole_roughness_m,extra.steel_roughness_m))
    wide=make('Возврат: вокруг бурильной трубы',Lw,d['brhl_diam_D'],d['out_diam_dril_pipe_d'],roughness=rough(d['brhl_diam_D'],d['out_diam_dril_pipe_d'],extra.borehole_roughness_m,extra.steel_roughness_m))
    recovery=extra.return_transition_pressure_recovery_pa
    # Early-event floor remains below the least possible valid interface state.
    floor=min(d['ATM_PRSSR_P0'],d['ATM_PRSSR_P0']-max(recovery,0.))
    if floor<=0:raise ValueError('Слишком большая заданная прибавка давления на переходе.')
    def evaluate(pin,early=True):
        c=replace(narrow,outlet_pressure_pa=floor)
        if extra.inlet_condition=='developed_inlet' and ms:
            c=replace(c,inlet_solids_velocity_m_s=m3_equilibrium_velocity(c,pin))
        nr=m3_integrate(c,pin,rtol,early)
        if not nr['reached_outlet']:return dict(residual=nr['residual'],reached_outlet=False)
        end=nr['rows'][-1];pjoin=end['pressure_pa']+recovery
        wc=replace(wide,inlet_solids_velocity_m_s=end['solids_velocity_m_s'])
        if pjoin<wc.outlet_pressure_pa:
            return dict(residual=pjoin-wc.outlet_pressure_pa,reached_outlet=False)
        wr=m3_integrate(wc,pjoin,rtol,early)
        return dict(residual=wr['residual'],reached_outlet=wr['reached_outlet'],narrow=nr,wide=wr,
                    narrow_case=c,wide_case=wc)
    # Bracket over valid trials only. A domain violation is never a root sign.
    lo=d['ATM_PRSSR_P0'];low=evaluate(lo)
    if low['residual']>0:raise M3DomainError('М3: давление возврата не заключено в скобку снизу.')
    hi=lo*1.5+1000.;upper=None
    for _ in range(30):
        try:trial=evaluate(hi)
        except M3DomainError as exc:
            invalid=hi;upper=None
            for _probe in range(36):
                mid=(lo+invalid)/2
                try:trial=evaluate(mid)
                except M3DomainError:invalid=mid;continue
                if trial['residual']>=0:hi=mid;upper=trial;break
                lo=mid
            if upper is None:raise M3DomainError('М3: не найдена допустимая скобка всего возврата: '+str(exc)) from exc
            break
        if trial['residual']>=0:upper=trial;break
        lo=hi;hi*=1.7
    else:raise M3DomainError('М3: нет скобки давления возврата.')
    root=m3_shoot_root(evaluate,lo,hi,pressure_tol,upper)
    hi=root['pressure_pa'];iteration=root['iterations']-1
    answer=evaluate(hi,False)
    if not answer['reached_outlet'] or abs(answer['residual'])>2*pressure_tol:raise M3DomainError('М3: не выполнено условие давления на устье.')
    return_sections=[];profile=[]
    for c,key,offset in [(answer['narrow_case'],'narrow',0.),(answer['wide_case'],'wide',Ln)]:
        part=answer[key]
        return_sections.append(dict(name=c.name,case=asdict(c),bottom_pressure_pa=part['rows'][0]['pressure_pa'],top_pressure_pa=part['rows'][-1]['pressure_pa'],
            diagnostics=m3_profile_diagnostics(c,part['rows']),rows=part['rows'],steps=part['steps']))
        profile.extend(dict(section=key,distance_from_bottom_m=r['z_m']+offset,**r) for r in part['rows'])
    before=answer['narrow']['rows'][-1];after=answer['wide']['rows'][0]
    transition=dict(pressure_before_pa=before['pressure_pa'],pressure_after_pa=after['pressure_pa'],
        prescribed_recovery_pa=recovery,solids_velocity_before_m_s=before['solids_velocity_m_s'],
        solids_velocity_after_m_s=after['solids_velocity_m_s'],
        matching='same us, independently conserved phase mass rates; static pressure jump specified by input; no inferred two-phase junction law')
    # Known return-bottom p plus externally supplied aggregate static interface jump.
    p=hi+extra.bit_pressure_drop_pa
    if p<=0:raise M3DomainError('М3: недопустимое давление за учётом стыков подачи.')
    supply=[];core_length=Ln*d['core_out_bk']/100
    components=[]
    if core_length>0:components.append(make('Подача: вокруг керна',core_length,d['in_diam_core_pipe_db'],d['core_diam_dk'],0.,rough(d['in_diam_core_pipe_db'],d['core_diam_dk'],extra.steel_roughness_m,extra.core_roughness_m)))
    if Ln-core_length>0:components.append(make('Подача: свободная колонковая труба',Ln-core_length,d['in_diam_core_pipe_db'],0.,0.))
    components.append(make('Подача: бурильная труба',Lw,d['in_diam_dril_pipe_d1'],0.,0.))
    supply_jumps=[]
    for c in components:
        if c.name=='Подача: свободная колонковая труба':
            jump=extra.core_interface_pressure_addition_pa
        elif c.name=='Подача: бурильная труба':
            jump=extra.barrel_pipe_interface_pressure_addition_pa
            if not any(x.name=='Подача: свободная колонковая труба' for x in components):jump+=extra.core_interface_pressure_addition_pa
            c=replace(c,extra_fanning=extra.connection_total_K*c.dh_m/(4*c.length_m))
        else:jump=0.
        p+=jump
        supply_jumps.append(jump)
        part=m3_gas_supply_section(p,c);supply.append(part);p=part['top_pressure_pa']
    supply_start=hi+extra.bit_pressure_drop_pa
    common_pressure_chain=[dict(name='Возврат вокруг бурильной трубы',increment_pa=after['pressure_pa']-d['ATM_PRSSR_P0']),
        dict(name='Обратный проход перехода',increment_pa=-recovery),
        dict(name='Возврат вокруг колонковой трубы',increment_pa=hi-before['pressure_pa']),
        dict(name='Заданный перепад коронки',increment_pa=extra.bit_pressure_drop_pa)]
    for x,jump in zip(supply,supply_jumps):
        if jump:common_pressure_chain.append(dict(name='Заданный скачок перед '+x['name'],increment_pa=jump))
        common_pressure_chain.append(dict(name=x['name'],increment_pa=x['top_pressure_pa']-x['bottom_pressure_pa']))
    closure=sum(x['increment_pa'] for x in common_pressure_chain)-(p-d['ATM_PRSSR_P0'])
    mass_g=max(abs(r['voidage']*r['gas_density_kg_m3']*r['gas_velocity_m_s']*(narrow.area_m2 if r['section']=='narrow' else wide.area_m2)/mg-1) for r in profile)
    mass_s=max(abs(r['solids_volume_fraction']*d['rock_density_ro']*r['solids_velocity_m_s']*(narrow.area_m2 if r['section']=='narrow' else wide.area_m2)/ms-1) for r in profile) if ms else 0.
    power=shared_power(d,gas,mg,p+d['receiver_margin_pa'])
    return dict(schema_version=3,model='M3 '+M3_VERSION,scope=M3_SCOPE,inputs=d,additional_inputs=asdict(extra),
        Q_reference_m3_min=q,mass_gas_kg_s=mg,mass_solids_kg_s=ms,
        outlet_pressure_pa=d['ATM_PRSSR_P0'],return_bottom_pressure_pa=hi,
        supply_bottom_pressure_pa=supply_start,surface_supply_pressure_pa=p,
        receiver_pressure_pa=p+d['receiver_margin_pa'],common_power=power,
        return_sections=return_sections,supply_sections=supply,return_profile=profile,transition=transition,
        pressure_budget=common_pressure_chain,pressure_budget_closure_pa=closure,
        outlet_residual_pa=answer['residual'],gas_mass_relative_error=mass_g,solids_mass_relative_error=mass_s,
        shooting_iterations=iteration+1,shooting_method='Illinois_with_bisection_safeguards',shooting_bracket_width_pa=root['bracket_width_pa'],numerical_tolerances=dict(rtol=rtol,pressure_boundary_pa=pressure_tol),minimum_radial_clearance_m=gap,
        particle_to_clearance_ratio=extra.particle_diameter_m/gap,maximum_particle_to_clearance_ratio=extra.maximum_particle_diameter_m/gap,
        physical_validation=False,author_example_reproduced=False,
        validation_status='conditional_stationary_solution; numerical_verification_separate_from_physical_validation',
        sources=M3_SOURCE_RECORDS,notes=[
        'Один массовый расход газа и шлама проходит оба восходящих участка; скорость шлама на стыке не сбрасывается.',
        'Падение/восстановление давления на резких переходах и коронке является входными данными, а не предсказанием этой модели.',
        'Нулевые скачки давления означают идеализированный сценарий. Их нельзя выдавать за измеренные сопротивления.',
        'Суммарный K соединений равномерно распределён по длине бурильной трубы как добавочное трение; это общее инженерное приближение.',
        'Шероховатость кольца усреднена по смоченному периметру; это явно заданное инженерное приближение.',
        'Проходимость сферической частицы проверена геометрически. Влияние стенок, формы и заклинивания при зазоре 2 мм экспериментально не проверено.',
        'Модель не подтверждает работу при любом паспорте: плотный, переходный, эксцентричный, влажный и противоточный режимы отклоняются.',
        'Стационарные уравнения и нормировка Yang проверяются; авторские кривые и нестационарный код не объявлены воспроизведёнными.'])


def m3_full_text(result):
    r=result
    lines=['М3 '+M3_VERSION+'; давления абсолютные.',
        'Q=%.8g м³/мин; mg=%.8g кг/с; ms=%.8g кг/с.'%(r['Q_reference_m3_min'],r['mass_gas_kg_s'],r['mass_solids_kg_s']),
        'Давление у забоя в возврате: %.7f МПа.'%(r['return_bottom_pressure_pa']/1e6),
        'Давление подачи у устья: %.7f МПа.'%(r['surface_supply_pressure_pa']/1e6),
        'Давление ресивера с паспортной добавкой: %.7f МПа.'%(r['receiver_pressure_pa']/1e6),
        'Невязка выхода %.6g Па; замыкание контура %.6g Па.'%(r['outlet_residual_pa'],r['pressure_budget_closure_pa']),
        'Невязки массовых расходов: газ %.3g; шлам %.3g.'%(r['gas_mass_relative_error'],r['solids_mass_relative_error']),
        'Дополнительные данные: '+r['additional_inputs']['data_status'],
        'Состав давления подачи (в обратном направлении от атмосферы):']
    lines.extend('%s: %+.6f кПа'%(x['name'],x['increment_pa']/1000) for x in r['pressure_budget'])
    lines.extend(r['notes'])
    return '\n'.join(lines)


def m3_export_full(result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    jp=unique_output(folder,'m3_full_result','json');cp=unique_output(folder,'m3_return_profile','csv');mp=unique_output(folder,'m3_full_report','md')
    jp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with cp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(result['return_profile'][0]));w.writeheader();w.writerows(result['return_profile'])
    mp.write_text(m3_full_text(result)+'\n\nИсточники:\n'+'\n'.join('- '+x['name']+': '+x['url'] for x in result['sources']),encoding='utf-8')
    return [jp,cp,mp]


def m3_input_passport(data):
    print('Все значения вводятся явно. Неизвестные потери нельзя выдавать за известный ноль.')
    params={}
    questions=[('particle_diameter_m','Средний эквивалентный диаметр частиц, мм',.001),
       ('maximum_particle_diameter_m','Максимальный диаметр частиц, мм',.001),
       ('inlet_solids_velocity_m_s','Скорость частиц у забоя, м/с',1.),
       ('viscosity_pa_s','Вязкость газа, Па*с',1.),
       ('steel_roughness_m','Шероховатость стали, мм',.001),
       ('borehole_roughness_m','Шероховатость скважины, мм',.001),
       ('core_roughness_m','Шероховатость керна, мм',.001),
       ('return_transition_pressure_recovery_pa','Прибавка статического давления на расширении возврата, Па (со знаком)',1.),
       ('bit_pressure_drop_pa','Перепад статического давления на коронке, Па (>=0)',1.),
       ('core_interface_pressure_addition_pa','Статический скачок перед участком вокруг керна в направлении газа, Па (со знаком)',1.),
       ('barrel_pipe_interface_pressure_addition_pa','Статический скачок между бурильной и колонковой трубой, Па (со знаком)',1.),
       ('connection_total_K','Сумма коэффициентов K соединений, отнесённых к скорости внутри бурильной трубы',1.)]
    for key,label,factor in questions:params[key]=number(input(label+': '))*factor
    value=input('Вход шлама: 1 заданная скорость; 2 локально установившийся режим: ').strip()
    if value not in ('1','2'):raise ValueError('Выберите 1 или 2.')
    params['inlet_condition']='specified_velocity' if value=='1' else 'developed_inlet'
    value=input('Yang: 1 только ветвь Adeleke с ug/Ut>1.5; 2 две исходные ветви Yang: ').strip()
    if value not in ('1','2'):raise ValueError('Выберите 1 или 2.')
    params['yang_regime']='source_high_only' if value=='1' else 'yang_two_branch'
    params['data_status']=input('Происхождение дополнительных данных (измерения/предположения): ').strip()
    return M3Passport(**params)


def m3_full_menu(data,flow):
    extra=None;last=None
    while True:
        print('\nМ3 v1.2: весь стационарный контур. Дополнительный паспорт хранится отдельно от общих данных.')
        print('1. Ввести дополнительные данные\n2. Показать данные\n3. Рассчитать весь контур\n4. Иллюстрация с ЯВНЫМИ предположениями\n5. Сохранить дополнительный паспорт\n6. Загрузить дополнительный паспорт\n7. Экспорт результата и профиля\n8. Опубликованные контрольные числа и статус\n9. Сравнить весь контур М1/М2/М3 на одинаковых условиях\n10. Анализ диапазона расхода\n0. Назад')
        try:
            key=input('Пункт М3: ').strip()
            if key=='0':return
            if key=='1':extra=m3_input_passport(data);last=None
            elif key=='2':print(json.dumps(asdict(extra) if extra else {'status':'Дополнительный паспорт ещё не введён'},ensure_ascii=False,indent=2))
            elif key in ('3','4'):
                if key=='4':extra=m3_example_passport(data)
                if extra is None:raise ValueError('Сначала введите дополнительный паспорт.')
                print('Статус данных: '+extra.data_status+'\nВычисление...')
                last=m3_full_circuit(data,flow,extra);print(m3_full_text(last))
            elif key=='5':
                if extra is None:raise ValueError('Нет дополнительного паспорта.')
                path=unique_output(Path(__file__).resolve().parent,'m3_passport','json')
                path.write_text(json.dumps(dict(schema_version=3,additional_inputs=asdict(extra)),ensure_ascii=False,indent=2),encoding='utf-8');print(str(path))
            elif key=='6':
                path=Path(input('JSON-файл: ').strip().strip(chr(34))).expanduser()
                payload=json.loads(path.read_text(encoding='utf-8-sig'))
                if payload.get('schema_version')!=3:raise ValueError('Неверная версия паспорта.')
                candidate=M3Passport(**payload['additional_inputs']);extra=candidate;last=None
            elif key=='7':
                if last is None:raise ValueError('Сначала выполните расчёт.')
                for path in m3_export_full(last,Path(__file__).resolve().parent):print('Сохранено: '+str(path))
            elif key=='8':print(json.dumps(m3_published_control(),ensure_ascii=False,indent=2))
            elif key=='9':
                if extra is None:raise ValueError('Сначала задайте дополнительный паспорт.')
                compared=m3_compare_full(data,flow,extra,last)
                print('Модель | Забой возврата, МПа | Подача устья, МПа | Ресивер, МПа')
                for row in compared['rows']:
                    if row.get('status')=='rejected':print(row['model']+': '+row['reason'])
                    else:print('%s | %.7f | %.7f | %.7f'%(row['model'],row['return_bottom_pressure_pa']/1e6,row['surface_supply_pressure_pa']/1e6,row['receiver_pressure_pa']/1e6))
                last=compared['M3']
                path=unique_output(Path(__file__).resolve().parent,'three_models_full_comparison','json')
                path.write_text(json.dumps(compared,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print('Сохранено: '+str(path))
            elif key=='10':
                if extra is None:raise ValueError('Сначала задайте дополнительный паспорт.')
                qmin=number(input('Минимальный Q, м³/мин: '));qmax=number(input('Максимальный Q: '));count=int(input('Число точек (2..101): '))
                scan=m3_scan_flow(data,extra,qmin,qmax,count)
                path=unique_output(Path(__file__).resolve().parent,'m3_flow_range','json')
                path.write_text(json.dumps(scan,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                print('Рассчитано %d/%d. '%(scan['computed'],scan['requested'])+scan['note']);print('Сохранено: '+str(path))
        except (ValueError,ArithmeticError,TypeError,KeyError,OSError) as exc:print('Расчёт/операция не принят: '+str(exc))
        except EOFError:return


_m3_old_published_control=_m3_base_published_control

def m3_published_control():
    r=_m3_old_published_control()
    r['published_optimum_table_5_6']=[dict(dp_um=x,solids_g_s=y,U_experiment_m_s=a,U_author_simulation_m_s=b)
        for x,y,a,b in [(346.5,15.14,6.2,7.2),(346.5,30.15,7.2,8.2),
          (713.5,15.11,9.5,9.4),(713.5,20.07,9.4,9.45),(852.5,25.09,9.98,11.2),
          (852.5,30.03,10.2,11.5),(1214.,25.25,11.8,12.8),(1214.,30.08,12.,13.06)]]
    r['source_inconsistencies']=[
        'Table5.4: mean852.5 um is below the stated sieve interval853-1200 um. Kept literally; no guessed replacement.',
        'Eq3.21: printed 10**-5 switch inconsistent with continuous TL sphere law. Continuous branch used explicitly.',
        'Eq3.12: density lacks phase index; bulk solids friction normalized independently from original Yang mass form.',
        'Eq3.20: sign of Di Felice exponent requires separate verification; Wen-Yu is explicitly used here.']
    r['status']='not_reproduced'
    r['project_version']=M3_VERSION
    return r


class Verification(Verification):
    def test_terminal_newton_against_independent_bisection(self):
        for rho,diam in [(1.2,.00035),(5.,.0012),(10.,.0001)]:
            u=m3_terminal_velocity(rho,3150,diam,1.8e-5)
            left,right=0.,9999*1.8e-5/(rho*diam)
            for _ in range(80):
                mid=(left+right)/2
                force=.75*m3_sphere_cd(rho*mid*diam/1.8e-5)*rho*mid*mid/diam
                if force>(3150-rho)*9.81:right=mid
                else:left=mid
            self.assertLess(abs(u/((left+right)/2)-1),1e-11)

    def test_optional_yang_low_branch_original_mass_normalization(self):
        c=replace(m3_demo_case(),particle_diameter_m=.0007135,yang_regime='yang_two_branch')
        st=m3_local(c,101325.,1.)
        self.assertEqual(st['yang_branch'],'low')
        alpha=st['solids_volume_fraction'];eps=st['voidage'];slip=st['slip_m_s'];ut=st['terminal_velocity_m_s']
        fp=.0410*alpha/eps**3*(alpha*ut/slip)**-1.021
        expected=fp*c.particle_density_kg_m3*alpha/(2*c.dh_m)
        self.assertAlmostEqual(st['solids_wall_N_m3'],expected)

    def test_equilibrium_inlet_is_boundary_condition(self):
        c=m3_demo_case();us=m3_equilibrium_velocity(c,101600.);st=m3_local(c,101600.,us)
        self.assertAlmostEqual(st['drag_N_m3']-st['solids_wall_N_m3']-st['solids_weight_N_m3'],0,delta=1e-8)

    def test_supply_gas_exact_zero_gravity_invariant(self):
        c=replace(m3_demo_case(),solids_mass_kg_s=0.,gravity=0.,length_m=100.)
        p0=200000.;r=m3_gas_supply_section(p0,c);p1=r['top_pressure_pa']
        re=c.gas_mass_kg_s*c.dh_m/(c.area_m2*c.viscosity_pa_s)
        f=m3_fanning(re,0);K=(c.gas_mass_kg_s/c.area_m2)**2*c.gas_R*c.temperature_k
        fun=lambda p:p*p-2*K*math.log(p)
        self.assertLess(abs((fun(p1)-fun(p0))/(4*f*K*c.length_m/c.dh_m)-1),1e-8)

    def test_passport_guards_maximum_size_and_eccentricity(self):
        x=m3_example_passport(default_data())
        with self.assertRaises(ValueError):replace(x,maximum_particle_diameter_m=.0001)
        with self.assertRaises(M3DomainError):replace(x,eccentricity_m=.0001)
        with self.assertRaises(ValueError):replace(x,viscosity_pa_s=float('nan'))
        with self.assertRaises(M3DomainError):m3_full_circuit(default_data(),None,replace(x,maximum_particle_diameter_m=.002))

    def test_full_loop_matching_and_conservation_short_geometry(self):
        d=dict(default_data(),dpth_h=3.,core_pipe_lngth_lk=.1)
        r=m3_full_circuit(d,None,m3_example_passport(d),rtol=1e-5)
        self.assertLess(abs(r['outlet_residual_pa']),.1)
        self.assertLess(abs(r['pressure_budget_closure_pa']),.1)
        self.assertLess(r['gas_mass_relative_error'],1e-12)
        self.assertLess(r['solids_mass_relative_error'],1e-12)
        t=r['transition'];self.assertEqual(t['solids_velocity_before_m_s'],t['solids_velocity_after_m_s'])
        self.assertAlmostEqual(t['pressure_after_pa']-t['pressure_before_pa'],t['prescribed_recovery_pa'])
        self.assertFalse(r['physical_validation']);self.assertFalse(r['author_example_reproduced'])

    def test_published_controls_are_not_our_own_results(self):
        r=m3_published_control();self.assertEqual(len(r['published_optimum_table_5_6']),8)
        self.assertEqual(r['status'],'not_reproduced');self.assertTrue(r['source_inconsistencies'])



def m3_controlled_homogeneous(data,flow,extra,model):
    """M1/M2 with EXACTLY the same prescribed interface data as the M3 run.

    Model names state whether a shared Chen diagnostic replaces native friction.
    All connectors use the SAME input total K, spread over the actual pipe length.
    """
    if model not in ('M1_AIR','M2_Nikuradse','M2_shared_Chen','M2_shared_Weymouth'):raise ValueError('Неизвестная модель сравнения.')
    d,gas,q,mg,ms,_=project_common(data,flow)
    if d['source_literal_mode'] or d['geometry_mode']!=1 or d['up_formula_variant']!=1 or abs(d['drilling_angle_alfa']-90)>1e-9:
        raise ValueError('Сравнение требует общего вертикального паспорта и варианта AIR=1.')
    channels,_=project_channels(d);p=d['ATM_PRSSR_P0'];rows=[];checkpoints={}
    has_empty=any(x['section'].name=='empty_core_barrel' for x in channels)
    first_down=True;mu=ms/mg;b=gas.g/(2*gas.R*d['well_temp_t'])
    for ch in channels:
        c=ch['section'];up=ch['flow']=='up';jump=0.
        if c.name=='annulus_core_barrel':jump=-extra.return_transition_pressure_recovery_pa
        elif not up and first_down:jump=extra.bit_pressure_drop_pa;first_down=False
        if c.name=='empty_core_barrel':jump+=extra.core_interface_pressure_addition_pa
        if c.name=='inside_drill_pipe':jump+=extra.barrel_pipe_interface_pressure_addition_pa+(0 if has_empty else extra.core_interface_pressure_addition_pa)
        p+=jump
        if p<=0:raise M3DomainError('Неположительное давление на стыке '+c.name)
        if c.name.startswith('annulus'):
            inner=d['out_diam_dril_pipe_d'] if c.name=='annulus_drill_pipe' else d['out_diam_core_pipe_dh']
            rough=(extra.steel_roughness_m*inner+extra.borehole_roughness_m*d['brhl_diam_D'])/(inner+d['brhl_diam_D'])
        elif c.name=='around_core':rough=(extra.steel_roughness_m*d['in_diam_core_pipe_db']+extra.core_roughness_m*d['core_diam_dk'])/(d['in_diam_core_pipe_db']+d['core_diam_dk'])
        else:rough=extra.steel_roughness_m
        if model in ('M1_AIR','M2_shared_Weymouth'):f=c.friction_darcy
        elif model=='M2_Nikuradse':f=nikuradse(rough,c.hydraulic_diameter_m)
        else:f=4*m3_fanning(mg*c.hydraulic_diameter_m/(c.area_m2*extra.viscosity_pa_s),rough/c.hydraulic_diameter_m)
        ksum=extra.connection_total_K if c.name=='inside_drill_pipe' else 0.
        start=p
        if model=='M1_AIR':
            n=int(d['air_substeps']);L=c.length_m/n
            a=mg**2*gas.R*d['well_temp_t']/(c.hydraulic_diameter_m*c.area_m2**2)
            for _ in range(n):
                if up:
                    den=1-(1+mu)*b*L
                    if den<=0:raise M3DomainError('Знаменатель AIR неположителен.')
                    p=(math.sqrt(p*p+f*(1+d['gast_koef_kr']*mu)*a*L)+p*(1+mu)*b*L)/den
                else:p=(math.sqrt(p*p+a*(f*L+ksum*c.hydraulic_diameter_m/n))-p*b*L)/(1+b*L)
        else:
            effective=replace(c,friction_darcy=f+ksum*c.hydraulic_diameter_m/c.length_m)
            p=section_pressure(p,effective,gas,mg,ms if up else 0.,ch['flow'],'top' if up else 'bottom')
        rows.append(dict(name=c.name,start_pressure_pa=start,end_pressure_pa=p,interface_increment_pa=jump,friction_darcy=f,connection_K=ksum,
            friction_regime=friction_regime_check(d['well_temp_t'],mg,c.area_m2,c.hydraulic_diameter_m,rough)))
        if c.name=='annulus_drill_pipe':checkpoints['return_interface_pressure_pa']=p
        if c.name=='annulus_core_barrel':checkpoints['return_bottom_pressure_pa']=p
    return dict(model=model,Q_reference_m3_min=q,mass_gas_kg_s=mg,mass_solids_kg_s=ms,surface_supply_pressure_pa=p,
        receiver_pressure_pa=p+d['receiver_margin_pa'],sections=rows,**checkpoints)


def m3_compare_full(data,flow,extra,m3_result=None):
    if isinstance(extra,dict):extra=M3Passport(**extra)
    result=m3_full_circuit(data,flow,extra) if m3_result is None else m3_result
    d,_,q,_,_,_=project_common(data,flow)
    if result['inputs']!=d or result['additional_inputs']!=asdict(extra) or result['Q_reference_m3_min']!=q:
        raise ValueError('Нельзя сравнивать результат с другим паспортом или расходом.')
    rows=[];details={}
    for name in ('M1_AIR','M2_Nikuradse','M2_shared_Chen'):
        try:
            rr=m3_controlled_homogeneous(data,flow,extra,name);details[name]=rr
            rows.append({k:rr[k] for k in ('model','return_bottom_pressure_pa','surface_supply_pressure_pa','receiver_pressure_pa')})
        except (ValueError,ArithmeticError) as exc:rows.append(dict(model=name,status='rejected',reason=str(exc)))
    rows.append(dict(model='M3_steady_two_velocity',return_bottom_pressure_pa=result['return_bottom_pressure_pa'],surface_supply_pressure_pa=result['surface_supply_pressure_pa'],receiver_pressure_pa=result['receiver_pressure_pa']))
    try:shared_weymouth=m3_controlled_homogeneous(data,flow,extra,'M2_shared_Weymouth')
    except (ValueError,ArithmeticError):shared_weymouth=None
    decomposition=None
    if shared_weymouth is not None and 'M1_AIR' in details and 'M2_shared_Chen' in details:
        a=details['M1_AIR']['surface_supply_pressure_pa'];b=shared_weymouth['surface_supply_pressure_pa'];c=details['M2_shared_Chen']['surface_supply_pressure_pa'];e=result['surface_supply_pressure_pa']
        decomposition=dict(M1_minus_M3_pa=a-e,M1_minus_M2_same_Weymouth_pa=a-b,
            M2_Weymouth_minus_M2_Chen_pa=b-c,M2_Chen_minus_M3_pa=c-e,
            note='Exact additive path through controlled variants; separates pressure/mass-loading formulation, gas wall law, and two-velocity plus acceleration changes. No fitted coefficient.')
    return dict(rows=rows,homogeneous_details=details,M3=result,decomposition=decomposition,M2_shared_Weymouth=shared_weymouth,
        comparability='Same geometry, phase mass rates, temperature, outlet p, prescribed interface jumps and connector total K. Native phase and wall closures remain different.',
        notes=['M2_shared_Chen is a controlled diagnostic adaptation, not the author Guo friction law.',
        'M1/M2 remain homogeneous; M3 has independent velocities and phase acceleration.',
        'Equal inputs do not guarantee validity for every regime. A rejected model has no comparable pressure.'])


class Verification(Verification):
    def test_controlled_comparison_recovers_old_homogeneous_algorithms(self):
        d=default_data();x=m3_example_passport(d)
        d.update(pipe_roughness_m=x.steel_roughness_m,borehole_roughness_m=x.borehole_roughness_m,core_roughness_m=x.core_roughness_m)
        for name,method,mode in [('M1_AIR','air',0),('M2_Nikuradse','guo',1),('M2_shared_Weymouth','guo',0)]:
            old=calculate(dict(d,guo_friction_mode=mode),None,method)
            new=m3_controlled_homogeneous(d,None,x,name)
            self.assertAlmostEqual(old['P4']*1e6,new['surface_supply_pressure_pa'],delta=1e-7)
            self.assertAlmostEqual(old['P2']*1e6,new['return_bottom_pressure_pa'],delta=1e-7)

    def test_interface_jumps_are_placed_at_their_own_nodes(self):
        d=default_data();x=replace(m3_example_passport(d),return_transition_pressure_recovery_pa=1000.,bit_pressure_drop_pa=2000.,core_interface_pressure_addition_pa=-100.,barrel_pipe_interface_pressure_addition_pa=300.)
        r=m3_controlled_homogeneous(d,None,x,'M1_AIR')
        jumps={row['name']:row['interface_increment_pa'] for row in r['sections']}
        self.assertEqual(jumps['annulus_core_barrel'],-1000.)
        self.assertEqual(jumps['around_core'],2000.)
        self.assertEqual(jumps['empty_core_barrel'],-100.)
        self.assertEqual(jumps['inside_drill_pipe'],300.)



def m3_scan_flow(data,extra,qmin,qmax,count):
    require_finite(qmin,qmax)
    if not 0<qmin<qmax or type(count) is not int or not 2<=count<=101:raise ValueError('Нужен положительный интервал и 2..101 точка.')
    if isinstance(extra,dict):extra=M3Passport(**extra)
    rows=[]
    for i in range(count):
        q=qmin+(qmax-qmin)*i/(count-1)
        row=dict(Q_reference_m3_min=q)
        try:
            r=m3_full_circuit(data,q,extra);c=m3_compare_full(data,q,extra,r)
            row.update(status='computed',return_bottom_pressure_pa=r['return_bottom_pressure_pa'],
                surface_supply_pressure_pa=r['surface_supply_pressure_pa'],receiver_pressure_pa=r['receiver_pressure_pa'],
                minimum_gas_to_terminal_ratio=min(s['diagnostics']['gas_to_terminal_ratio_min'] for s in r['return_sections']) if r['mass_solids_kg_s'] else None,
                maximum_solids_fraction=max(s['diagnostics']['solids_volume_fraction_max'] for s in r['return_sections']))
            for item in c['rows']:
                if item['model']!='M3_steady_two_velocity':row[item['model']+'_surface_supply_pa']=item.get('surface_supply_pressure_pa')
        except (ValueError,ArithmeticError) as exc:row.update(status='rejected',reason=str(exc))
        rows.append(row);print('М3: точка %d/%d, Q=%.6g, %s'%(i+1,count,q,row['status']))
    valid=[r for r in rows if r['status']=='computed']
    return dict(rows=rows,computed=len(valid),requested=count,
        minimum_sampled_pressure_point=min(valid,key=lambda r:r['surface_supply_pressure_pa']) if valid else None,
        note='Конечная сетка. Минимум по точкам не является доказанным минимумом FCC или гарантией выноса. Отказ не является нулевым давлением.')


class Verification(Verification):
    def test_additional_passport_json_roundtrip(self):
        x=m3_example_passport(default_data());self.assertEqual(x,M3Passport(**json.loads(json.dumps(asdict(x),allow_nan=False))))
        self.assertGreater(x.connection_total_K,0)

    def test_domain_failure_preserves_physical_reason(self):
        x=replace(m3_demo_case(),gas_mass_kg_s=1e-7)
        with self.assertRaisesRegex(M3DomainError,'Re < 4000'):m3_solve(x)

class Verification(Verification):
    def test_sensitive_full_loop_checks_outlet_residual_not_only_bracket_width(self):
        d=dict(default_data(),dpth_h=100.,mech_spd_vm=6.)
        r=m3_full_circuit(d,6.,m3_example_passport(d))
        self.assertLess(abs(r['outlet_residual_pa']),.02)
        self.assertLess(abs(r['pressure_budget_closure_pa']),.1)


class Verification(Verification):
    def test_optional_low_yang_branch_solves_a_whole_section(self):
        c=replace(m3_demo_case(),particle_diameter_m=.0007135,yang_regime='yang_two_branch')
        r=m3_solve(c)
        self.assertTrue(r['reached_outlet']);self.assertLess(abs(r['residual']),.1)
        self.assertEqual(r['diagnostics']['yang_branches'],['low'])
        self.assertGreater(r['pressure_drop_pa'],0)



# Published evidence and uncertainty tools. All numerical fields are SI.
# No probability distribution is assigned to unknown passport parameters.
TEMPLE_EVIDENCE_SOURCE='https://etda.libraries.psu.edu/catalog/26245'


def m3_temple_evidence():
    """Independent transcriptions retain contradictory tables as separate records.

    Pressure-curve points are NOT reconstructed from axis labels or our own model.
    Appendix D optimum values differ from Chapter 5; neither is silently corrected.
    """
    table56=[(346.5,15.14,6.2,7.2),(346.5,30.15,7.2,8.2),
        (713.5,15.11,9.5,9.4),(713.5,20.07,9.4,9.45),
        (852.5,25.09,9.98,11.2),(852.5,30.03,10.2,11.5),
        (1214.,25.25,11.8,12.8),(1214.,30.08,12.,13.06)]
    appendix=[(3.47e-4,.015,3.0,6.8),(3.47e-4,.030,3.0,7.2),
        (7.14e-4,.015,5.92,9.0),(7.14e-4,.020,5.92,9.4),
        (8.53e-4,.025,6.9,9.9),(8.53e-4,.030,6.9,10.2),
        (1.21e-3,.025,9.18,11.8),(1.21e-3,.030,9.18,11.9)]
    error_by_diameter={346.5:57.7,713.5:56.9,852.5:110.,1214.:113.}
    series=[]
    for i,(dp,ms,Ue,Ua) in enumerate(table56):
        a=appendix[i]
        series.append(dict(series_id='Temple_'+str(i+1),particle_diameter_m=dp*1e-6,
            solids_mass_kg_s=ms*.001,optimum_experiment_table56_m_s=Ue,
            optimum_author_simulation_table56_m_s=Ua,
            appendix_D1=dict(particle_diameter_m=a[0],solids_mass_kg_s=a[1],terminal_velocity_m_s=a[2],optimum_m_s=a[3]),
            reported_pressure_error_pa=error_by_diameter[dp],
            pressure_error_is_standard_deviation=False,pressure_curve_points=[],
            pressure_curve_status='not_digitized_no_accessible_graph_bytes',
            source_location='Table5.6 p98; TableD.1 p162; pressure errors pp95-96',
            optimum_table_conflict=abs(Ue-a[3])>.005))
    return dict(source=TEMPLE_EVIDENCE_SOURCE,open_text='https://scispace.com/pdf/pneumatic-transport-modeling-of-air-gas-drilling-hydraulics-1curcuqn82.pdf',
        geometry=dict(outer_diameter_m=.0779,inner_diameter_m=.0422,length_m=2.74,
            temperature_k=288.1,viscosity_pa_s=1.760e-5,particle_density_kg_m3=3150.,outlet_pressure_pa=101325.),
        geometry_note='Outlet atmospheric p is a project scenario; published temperature and geometry are transcribed.',
        series=series,experimental_pressure_validation=False,
        notes=['Source Table5.6 and AppendixD.1 disagree for some optimum velocities; records are kept separate.',
        'Pressure measurement error bounds are not stated standard deviations and are not converted to 95% intervals.',
        'No pressure points have been invented from graph axis ranges.'])


def m3_terminal_published_checks():
    e=m3_temple_evidence();rho=101325/(287*288.1);rows=[]
    for item in e['series'][::2]:
        published=item['appendix_D1']['terminal_velocity_m_s']
        # Keep Chapter5 nominal particle diameters, expose AppendixD rounding.
        predicted=m3_terminal_velocity(rho,3150.,item['particle_diameter_m'],1.760e-5)
        rows.append(dict(particle_diameter_m=item['particle_diameter_m'],
            appendix_diameter_m=item['appendix_D1']['particle_diameter_m'],
            published_terminal_velocity_m_s=published,calculated_terminal_velocity_m_s=predicted,
            relative_difference_percent=100*(predicted/published-1)))
    return dict(rows=rows,max_absolute_difference_percent=max(abs(x['relative_difference_percent']) for x in rows),
        status='published_derived_quantity_agreement_not_experimental_pressure_validation',
        note='The source does not establish that its terminal velocities were measured independently. Rounding and reference gas state matter.')


def m3_temple_case(series,U,wall_roughness_m=0.,inlet_solids_velocity_m_s=1.,yang_regime='source_high_only'):
    require_finite(U,wall_roughness_m,inlet_solids_velocity_m_s)
    if U<=0:raise ValueError('Положительная скорость газа обязательна.')
    g=m3_temple_evidence()['geometry'];A=math.pi/4*(g['outer_diameter_m']**2-g['inner_diameter_m']**2)
    return M3Case(series['series_id']+'; boundary scenario',g['length_m'],g['outer_diameter_m'],g['inner_diameter_m'],
        g['outlet_pressure_pa']/(287*g['temperature_k'])*U*A,series['solids_mass_kg_s'],series['particle_diameter_m'],
        g['particle_density_kg_m3'],inlet_solids_velocity_m_s,g['outlet_pressure_pa'],g['temperature_k'],g['viscosity_pa_s'],
        wall_roughness_m,yang_regime=yang_regime)


def m3_temple_prediction(series,U,inlet_condition='specified_velocity',yang_regime='source_high_only',
                         wall_roughness_m=0.,inlet_solids_velocity_m_s=1.,rtol=2e-6):
    c=m3_temple_case(series,U,wall_roughness_m,inlet_solids_velocity_m_s,yang_regime)
    r=m3_solve(c,rtol,inlet_condition=inlet_condition);first,last=r['rows'][0],r['rows'][-1]
    return dict(U_outlet_superficial_m_s=U,U_inlet_superficial_m_s=U*c.outlet_pressure_pa/r['inlet_pressure_pa'],
        delta_p_pa=r['pressure_drop_pa'],inlet_solids_velocity_m_s=first['solids_velocity_m_s'],
        outlet_solids_velocity_m_s=last['solids_velocity_m_s'],
        minimum_gas_to_terminal_ratio=r['diagnostics']['gas_to_terminal_ratio_min'],
        yang_extrapolated=bool(r['diagnostics']['yang_extrapolated_nodes']),
        inlet_condition=inlet_condition,yang_regime=yang_regime,
        author_reproduction=False,experimental_pressure_validation=False)


def m3_pressure_error_metrics(points,predictions):
    """Absolute errors and deterministic coverage of reported experimental bounds.

    Points must carry source and digitization provenance. Experimental uncertainty
    and digitization error bounds are added conservatively, not treated as sigma.
    Never use agreement with computed points as experimental validation.
    """
    if not points:return dict(status='no_experimental_pressure_points',comparable_points=0,
                              experimental_validation=False)
    if len(points)!=len(predictions):raise ValueError('Число измерений и прогнозов различается.')
    rows=[]
    for p,y in zip(points,predictions):
        if p.get('kind')!='experimental' or not p.get('source') or not p.get('extraction_method'):
            raise ValueError('Нужны экспериментальное происхождение, источник и способ извлечения каждой точки.')
        expected=p['delta_p_pa'];ue=p['reported_error_bound_pa'];ud=p['digitization_error_bound_pa']
        require_finite(expected,ue,ud,y)
        if expected<=0 or min(ue,ud)<0:raise ValueError('Неверное давление или граница погрешности.')
        error=y-expected;bound=ue+ud
        rows.append(dict(measured_pa=expected,predicted_pa=y,error_pa=error,
            absolute_relative_error_percent=100*abs(error)/expected,
            comparison_bound_pa=bound,inside_reported_bounds=abs(error)<=bound,
            excess_outside_bounds_pa=max(0.,abs(error)-bound)))
    n=len(rows)
    return dict(status='compared_with_reported_bounds',comparable_points=n,rows=rows,
        MAE_pa=sum(abs(x['error_pa']) for x in rows)/n,
        RMSE_pa=math.sqrt(sum(x['error_pa']**2 for x in rows)/n),
        bias_pa=sum(x['error_pa'] for x in rows)/n,
        inside_reported_bounds_count=sum(x['inside_reported_bounds'] for x in rows),
        experimental_validation='limited_to_input_points_and_boundary_assumptions',
        note='Coverage is a deterministic count within supplied error bounds, not a confidence probability or universal accuracy.')


def m3_parameter_probe(data,flow,extra,target,value):
    """One explicit input perturbation. Keep the chosen reference flow fixed."""
    d=dict(data);x=extra
    if target=='Q_reference_m3_min':flow=value
    elif target=='radial_clearance_m':
        if value<=0:raise ValueError('Зазор должен быть положительным.')
        d['out_diam_core_pipe_dh']=d['brhl_diam_D']-2*value
    elif target in asdict(x):
        x=replace(x,**{target:value})
    elif target in d:d[target]=value
    else:raise ValueError('Неизвестный параметр: '+target)
    return d,flow,x


def m3_sensitivity_specs(data,flow,extra):
    d,_,q,_,_,_=project_common(data,flow);gap=(d['brhl_diam_D']-d['out_diam_core_pipe_dh'])/2
    specs=[]
    for key in ('steel_roughness_m','borehole_roughness_m','core_roughness_m','particle_diameter_m',
                'inlet_solids_velocity_m_s','viscosity_pa_s','connection_total_K'):
        value=getattr(extra,key)
        if value>0:specs.append(dict(parameter=key,base=value,levels=[.9*value,1.1*value],
            interpretation='local finite perturbation +/-10%; numerical sensitivity step, NOT a measured uncertainty bound'))
    specs.append(dict(parameter='radial_clearance_m',base=gap,levels=[.95*gap,1.05*gap],
        interpretation='local geometric perturbation +/-5%; NOT a manufacturing tolerance'))
    specs.append(dict(parameter='Q_reference_m3_min',base=q,levels=[.95*q,1.05*q],
        interpretation='flow perturbation +/-5%; NOT an instrument accuracy'))
    for key in ('bit_pressure_drop_pa','return_transition_pressure_recovery_pa',
                'core_interface_pressure_addition_pa','barrel_pipe_interface_pressure_addition_pa'):
        value=getattr(extra,key);specs.append(dict(parameter=key,base=value,levels=[value,value+1000.],
            interpretation='one-sided gain for an explicit +1000 Pa perturbation; NOT a known pressure-loss range'))
    return specs


def m3_sensitivity_screen(data,flow,extra,rtol=2e-7,progress=False):
    if isinstance(extra,dict):extra=M3Passport(**extra)
    d,_,q,_,_,_=project_common(data,flow);base=m3_full_circuit(d,q,extra,rtol);rows=[]
    p0=base['surface_supply_pressure_pa'];specs=m3_sensitivity_specs(d,q,extra)
    for i,spec in enumerate(specs):
        values=[]
        for value in spec['levels']:
            if value==spec['base']:r=base
            else:
                try:
                    dd,qq,xx=m3_parameter_probe(d,q,extra,spec['parameter'],value)
                    r=m3_full_circuit(dd,qq,xx,rtol)
                except (ValueError,ArithmeticError) as exc:
                    values.append(dict(value=value,status='rejected',reason=str(exc)));continue
            values.append(dict(value=value,status='computed',pressure_pa=r['surface_supply_pressure_pa'],
                relative_change_percent=100*(r['surface_supply_pressure_pa']/p0-1)))
        row=dict(**spec,results=values)
        if all(x['status']=='computed' for x in values):
            slope=(values[1]['pressure_pa']-values[0]['pressure_pa'])/(spec['levels'][1]-spec['levels'][0])
            row.update(local_gain=slope,dimensionless_elasticity=slope*spec['base']/p0 if spec['base'] else None,
                sampled_pressure_span_pa=abs(values[1]['pressure_pa']-values[0]['pressure_pa']))
        rows.append(row)
        if progress:print('Чувствительность %d/%d: %s'%(i+1,len(specs),spec['parameter']))
    return dict(baseline_pressure_pa=p0,rows=rows,rtol=rtol,
        interpretation='Local sensitivity screen only. Probe sizes are experimental-design choices, not empirically established uncertainty bounds.',
        confidence_level=None,physical_uncertainty_interval_available=False)


def m3_interval_audit(data,flow,extra,bounds,rtol=M3_DEFAULT_RTOL,progress=False):
    """Deterministic Cartesian grid at lower/mid/upper values, at most 243 runs.

    Each bound carries provenance and evidence status. No distribution, percentile,
    or confidence level is assigned. Interior extrema are not certified by a grid.
    """
    import itertools
    if isinstance(extra,dict):extra=M3Passport(**extra)
    if not isinstance(bounds,list) or not 1<=len(bounds)<=5:raise ValueError('Задайте от 1 до 5 интервалов.')
    parameters=[];levels=[];seen=set()
    for item in bounds:
        key=item['parameter'];lo=item['lower'];hi=item['upper'];require_finite(lo,hi)
        if key in seen or not lo<hi:raise ValueError('Повтор параметра или неверные границы.')
        if not item.get('provenance') or item.get('evidence_status') not in ('measured_bound','published_applicable_bound','hypothetical_scenario'):
            raise ValueError('У каждого интервала нужны происхождение и статус обоснования.')
        seen.add(key);parameters.append(key);levels.append([lo,(lo+hi)/2,hi])
    d,_,q,_,_,_=project_common(data,flow);rows=[]
    for i,values in enumerate(itertools.product(*levels)):
        row=dict(parameters=dict(zip(parameters,values)))
        try:
            dd,qq,xx=m3_apply_parameter_map(d,q,extra,row['parameters'])
            r=m3_full_circuit(dd,qq,xx,rtol)
            row.update(status='computed',pressure_pa=r['surface_supply_pressure_pa'],
                outlet_residual_pa=r['outlet_residual_pa'],maximum_particle_to_clearance_ratio=r['maximum_particle_to_clearance_ratio'])
        except (ValueError,ArithmeticError) as exc:row.update(status='rejected',reason=str(exc))
        rows.append(row)
        if progress:print('Интервальная сетка %d/%d: %s'%(i+1,3**len(bounds),row['status']))
    valid=[r for r in rows if r['status']=='computed'];all_evidenced=all(x['evidence_status']!='hypothetical_scenario' for x in bounds)
    return dict(bounds=bounds,rows=rows,requested=len(rows),computed=len(valid),rejected=len(rows)-len(valid),
        sampled_min_pressure_pa=min(x['pressure_pa'] for x in valid) if valid else None,
        sampled_max_pressure_pa=max(x['pressure_pa'] for x in valid) if valid else None,
        evidence_based_input_bounds=all_evidenced,confidence_level=None,guaranteed_continuous_enclosure=False,
        status='conditional_sampled_envelope' if valid else 'no_valid_results',
        note='Результат относится только к заданным интервалам и узлам. Это не 95%-ная точность, не сертифицированная оценка экстремумов между узлами и не экспериментальная валидация.')


def m3_evidence_menu(data,flow):
    while True:
        print('\nМ3: экспериментальные источники и неопределённость')
        print('1. Опубликованные данные и противоречия\n2. Контроль терминальных скоростей\n3. Чувствительность по дополнительному паспорту\n4. Интервальная сетка из JSON\n5. Экспериментальные кривые Narimatsu: расчёт и экспорт\n0. Назад')
        try:
            key=input('Пункт: ').strip()
            if key=='0':return
            if key=='1':print(json.dumps(m3_temple_evidence(),ensure_ascii=False,indent=2))
            elif key=='2':print(json.dumps(m3_terminal_published_checks(),ensure_ascii=False,indent=2))
            elif key=='5':
                result=m3_narimatsu_benchmark()
                print('Условное сравнение с экспериментом, без подбора коэффициентов. Газовое состояние и шероховатость заданы как сценарий.')
                for group in ('air_metrics','solids_metrics'):
                    print(group+': '+json.dumps(result[group],ensure_ascii=False))
                path=unique_output(Path(__file__).resolve().parent,'m3_experimental_curves','json')
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print('Сохранено: '+str(path))
            elif key in ('3','4'):
                payload=json.loads(Path(input('Файл дополнительного паспорта М3 JSON: ').strip().strip(chr(34))).read_text(encoding='utf-8-sig'))
                extra=M3Passport(**payload['additional_inputs'])
                if key=='3':result=m3_sensitivity_screen(data,flow,extra,progress=True)
                else:
                    bounds=json.loads(Path(input('Файл интервалов JSON: ').strip().strip(chr(34))).read_text(encoding='utf-8-sig'))
                    result=m3_interval_audit(data,flow,extra,bounds['bounds'],progress=True)
                path=unique_output(Path(__file__).resolve().parent,'m3_uncertainty_analysis','json')
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print('Сохранено: '+str(path))
        except (ValueError,ArithmeticError,TypeError,KeyError,OSError) as exc:print('Анализ не принят: '+str(exc))
        except EOFError:return


class Verification(Verification):
    def test_published_terminal_velocity_controls(self):
        r=m3_terminal_published_checks();self.assertEqual(len(r['rows']),4)
        self.assertLess(r['max_absolute_difference_percent'],.5)
        self.assertIn('not_experimental',r['status'])

    def test_conflicting_source_tables_are_preserved(self):
        r=m3_temple_evidence();self.assertEqual(len(r['series']),8)
        a=r['series'][0];self.assertEqual(a['optimum_experiment_table56_m_s'],6.2)
        self.assertEqual(a['appendix_D1']['optimum_m_s'],6.8)
        self.assertTrue(a['optimum_table_conflict']);self.assertFalse(r['experimental_pressure_validation'])
        self.assertTrue(all(not x['pressure_curve_points'] for x in r['series']))

    def test_pressure_metrics_require_experimental_provenance(self):
        self.assertEqual(m3_pressure_error_metrics([],[])['status'],'no_experimental_pressure_points')
        with self.assertRaises(ValueError):m3_pressure_error_metrics([dict(delta_p_pa=100.)],[100.])
        points=[dict(kind='experimental',source='test fixture',extraction_method='test fixture',delta_p_pa=100.,reported_error_bound_pa=10.,digitization_error_bound_pa=2.)]
        a=m3_pressure_error_metrics(points,[111.]);self.assertEqual(a['inside_reported_bounds_count'],1)
        b=m3_pressure_error_metrics(points,[113.]);self.assertEqual(b['inside_reported_bounds_count'],0)
        self.assertEqual(b['rows'][0]['excess_outside_bounds_pa'],1.)

    def test_literal_yang_extrapolation_is_flagged(self):
        c=replace(m3_demo_case(),particle_diameter_m=.001214,yang_regime='adeleke_literal_high',gas_mass_kg_s=m3_demo_case().gas_mass_kg_s*12/8)
        st=m3_local(c,101325.,1.);self.assertTrue(st['yang_extrapolated'])
        with self.assertRaises(M3DomainError):m3_local(replace(c,yang_regime='source_high_only'),101325.,1.)

    def test_invalid_uncertainty_bounds_rejected_before_calculation(self):
        d=default_data();x=m3_example_passport(d)
        with self.assertRaises(ValueError):m3_interval_audit(d,None,x,[])
        with self.assertRaises(ValueError):m3_interval_audit(d,None,x,[dict(parameter='steel_roughness_m',lower=1e-5,upper=2e-5)])


def m3_apply_parameter_map(data,flow,extra,parameters):
    """Apply an interval point atomically; never alter a sampled maximum size."""
    d=dict(data);values=asdict(extra);q=flow
    for key,value in parameters.items():
        require_finite(value)
        if key=='Q_reference_m3_min':q=value
        elif key=='radial_clearance_m':
            if value<=0:raise ValueError('Зазор должен быть положительным.')
            d['out_diam_core_pipe_dh']=d['brhl_diam_D']-2*value
        elif key in values:values[key]=value
        elif key in d:d[key]=value
        else:raise ValueError('Неизвестный параметр интервала: '+key)
    return d,q,M3Passport(**values)

class Verification(Verification):
    def test_interval_point_does_not_silently_increase_maximum_particle(self):
        d=default_data();x=m3_example_passport(d)
        with self.assertRaises(ValueError):
            m3_parameter_probe(d,None,x,'particle_diameter_m',.0005)
        with self.assertRaises(ValueError):
            m3_apply_parameter_map(d,None,x,dict(particle_diameter_m=.0005,maximum_particle_diameter_m=.0004))
        dd,q,xx=m3_apply_parameter_map(d,None,x,dict(particle_diameter_m=.0005,maximum_particle_diameter_m=.0006))
        self.assertEqual(xx.particle_diameter_m,.0005);self.assertEqual(xx.maximum_particle_diameter_m,.0006)

    def test_single_section_developed_inlet_is_applied_at_solved_pressure(self):
        c=m3_demo_case();r=m3_solve(c,inlet_condition='developed_inlet')
        us=m3_equilibrium_velocity(c,r['inlet_pressure_pa'])
        self.assertAlmostEqual(us,r['rows'][0]['solids_velocity_m_s'],delta=1e-9)
        self.assertLess(abs(r['residual']),.1)


class Verification(Verification):
    def test_roughness_probes_target_effective_additional_passport(self):
        d=default_data();x=m3_example_passport(d)
        for key in ('borehole_roughness_m','core_roughness_m'):
            dd,q,xx=m3_parameter_probe(d,None,x,key,.0002)
            self.assertEqual(getattr(xx,key),.0002)
            self.assertEqual(dd[key],d[key])
            dd,q,xx=m3_apply_parameter_map(d,None,x,{key:.0002})
            self.assertEqual(getattr(xx,key),.0002)

    def test_increasing_borehole_roughness_increases_full_pressure(self):
        d=dict(default_data(),dpth_h=3.,core_pipe_lngth_lk=.1);x=m3_example_passport(d)
        a=m3_full_circuit(d,None,x)
        dd,q,xx=m3_parameter_probe(d,None,x,'borehole_roughness_m',.0002)
        b=m3_full_circuit(dd,q,xx)
        self.assertGreater(b['surface_supply_pressure_pa'],a['surface_supply_pressure_pa'])



# Independent experimental pressure-gradient evidence. All digitized data are
# embedded here so the operational program remains a single stdlib Python file.
def m3_narimatsu_evidence():
    """Selected unambiguous observed markers, not values sampled from a fit.

    Manual centers checked on a 4x image; hollow air markers checked from enclosed
    white components. Original image 637x515 pixels. +/-2px includes center and
    axis placement resolution; NOT an instrument uncertainty or standard error.
    Air symbols repeat across panels: retain panel (a) only.
    """
    source='https://www.scielo.br/j/bjce/a/nwKM58MWFpjBrcvRsM6G3Wf/'
    axes={'a': {'x0': 44.0, 'x40': 310.0, 'y0': 165.0, 'y400': 8.0}, 'b': {'x0': 362.0, 'x40': 627.0, 'y0': 166.0, 'y400': 11.0}, 'c': {'x0': 197.0, 'x40': 467.0, 'y0': 404.0, 'y400': 246.0}}
    series={'air': {'panel': 'a', 'ms': 0.0, 'dp': 0.001, 'pixels': [(121.38, 148.08), (138.69, 140.88), (142.42, 137.83), (147.6, 135.8), (167.23, 122.77), (170.77, 119.23), (188.33, 108.67), (192.38, 105.12), (205.5, 92.5), (215.4, 85.4), (224.72, 78.89), (234.72, 70.89), (249.11, 55.28), (253.6, 49.4), (257.67, 44.89)]}, 'glass_1mm_0041': {'panel': 'a', 'ms': 0.041, 'dp': 0.001, 'pixels': [(117.8, 117.5), (147.5, 117.5), (170.5, 109.5), (193.28, 97.59), (216.72, 90.41)]}, 'glass_205mm_0115': {'panel': 'b', 'ms': 0.115, 'dp': 0.00205, 'pixels': [(445.62, 40.85), (455.5, 65.5), (466.34, 74), (488.21, 70.45), (498.42, 65.94), (520.73, 54.45), (534.41, 52.28), (547.89, 41.89)]}, 'glass_285mm_0068': {'panel': 'c', 'ms': 0.068, 'dp': 0.00285, 'pixels': [(301, 328), (312.13, 337), (341.2, 341.16), (357.04, 329.08), (369.87, 323.04)]}, 'glass_285mm_0086': {'panel': 'c', 'ms': 0.086, 'dp': 0.00285, 'pixels': [(324.29, 326.24), (339.28, 322.67), (352.5, 318.5), (365.5, 313.5)]}}
    points=[]
    for name,s in series.items():
        a=axes[s['panel']];dx=40/(a['x40']-a['x0']);dy=400/(a['y0']-a['y400'])
        for i,(x,y) in enumerate(s['pixels'],1):
            U=(x-a['x0'])*dx
            points.append(dict(point_id=name+'_'+str(i).zfill(2),panel=s['panel'],series_id=name,
                kind='experimental',source=source,source_figure='4('+s['panel']+')',
                extraction_method='manual marker center from original 637x515 GIF; ambiguous/overlapping markers excluded',
                pixel_x=x,pixel_y=y,digitization_bound_px=2.,superficial_velocity_m_s=U,
                gradient_pa_m=(a['y0']-y)*dy,velocity_digitization_bound_m_s=2*dx,
                gradient_digitization_bound_pa_m=2*dy,solids_mass_kg_s=s['ms'],
                particle_diameter_m=s['dp'],particle_density_kg_m3=2500.,
                measurement_error_bound_pa_m=None,measurement_velocity_error_bound_m_s=None,
                primary_curve_subset=s['ms']==0 or U-2*dx>{'a':14.,'b':17.2,'c':20.}[s['panel']]))
    return dict(source=source,doi='10.1590/S0104-66322001000300002',
        citation='Narimatsu CP & Ferreira MC, 2001, Braz J Chem Eng 18(3), Fig.4',
        image_size_px=[637,515],figure_sha256='cff16bbf833aaadc37b69183ca53e0ca6c2de9bfaeed73878915407575fadc13',axis_calibration=axes,
        figure_url='https://minio.scielo.br/documentstore/1678-4383/nwKM58MWFpjBrcvRsM6G3Wf/83872410f1d219a2ccb38a213391565be6153938.gif',
        geometry=dict(outer_diameter_m=.0534,inner_diameter_m=0.,total_length_m=3.,developed_above_z_m=.83),
        table1_terminal_velocities_literal_m_s={'1.00mm':7.45,'2.05mm':8.60,'2.85mm':8.83,'3.68mm':8.95},
        points=points,selection='Selected identifiable markers only, not all plotted measurements; dilute-side cut uses upper reported Umin per material, independent of fit.',
        source_data_limitations=['Absolute gas state, effective roughness and pressure-gradient measurement error are not supplied by the accessible source.',
            'Reported <=5.1% mean relative solids-flow deviations are NOT hard bounds or standard deviations.',
            'The 2.05m trap section is NOT stated to be the exact pressure-fit span. It is not used as that span here.',
            'Printed Table1 terminal velocities are estimated quantities and disagree with sphere-balance estimates for larger glass beads; neither table is silently corrected.'])


def m3_narimatsu_local(point,superficial_velocity=None,temperature_k=293.15,
                       pressure_pa=101325.,roughness_m=.00009144):
    """Conditional local developed static-pressure gradient, not entrance loss.

    Set us from Fdrag=Fwall+Fweight at the local gas state, then evaluate the
    implemented compressible differential balances. us'=0 at this point only;
    finite-length acceleration can be checked separately with m3_solve.
    Measured superficial U is referenced to the assumed LOCAL gas state.
    """
    U=point['superficial_velocity_m_s'] if superficial_velocity is None else superficial_velocity
    require_finite(U,temperature_k,pressure_pa,roughness_m)
    if U<=0:raise ValueError('Положительная поверхностная скорость обязательна.')
    T=temperature_k;rho=pressure_pa/(287*T);area=math.pi/4*.0534**2
    mu=1.716e-5*(T/273.)**1.5*(273.+111)/(T+111)
    c=M3Case('Narimatsu Fig4; conditional developed section',.1,.0534,0.,rho*U*area,
        point['solids_mass_kg_s'],point['particle_diameter_m'],2500.,1.,pressure_pa,T,mu,roughness_m)
    us=m3_equilibrium_velocity(c,pressure_pa);c=replace(c,inlet_solids_velocity_m_s=us)
    st=m3_local(c,pressure_pa,us);dp,dus=m3_derivative(c,pressure_pa,us)
    return dict(gradient_pa_m=-dp,solids_velocity_m_s=us,gas_velocity_m_s=st['gas_velocity_m_s'],
        voidage=st['voidage'],gas_to_terminal_ratio=st['gas_to_terminal_ratio'],
        particle_Re=st['particle_Re'],gas_Re=st['gas_Re'],yang_branch=st['yang_branch'],
        yang_extrapolated=st['yang_extrapolated'],equilibrium_dus_dz=dus,
        forces_pa_m={key:st[key] for key in ('gas_wall_N_m3','gas_weight_N_m3','drag_N_m3','solids_wall_N_m3','solids_weight_N_m3')},
        case=asdict(c))


def m3_gradient_metrics(rows):
    valid=[r for r in rows if r.get('status')=='computed' and r.get('primary_curve_subset')]
    if not valid:return dict(count=0,status='no_eligible_points')
    err=[r['prediction']['gradient_pa_m']-r['gradient_pa_m'] for r in valid]
    return dict(count=len(valid),MAE_pa_m=sum(abs(e) for e in err)/len(err),
        RMSE_pa_m=math.sqrt(sum(e*e for e in err)/len(err)),bias_pa_m=sum(err)/len(err),
        maximum_absolute_relative_difference_percent=max(100*abs(e)/r['gradient_pa_m'] for e,r in zip(err,valid)),
        MAPE_percent=sum(100*abs(e)/r['gradient_pa_m'] for e,r in zip(err,valid))/len(valid),
        confidence_level=None,interpretation='Descriptive discrepancies on selected points at assumed gas state and wall roughness; not universal accuracy or a confidence interval.')


def m3_interpolate_air(points,U):
    """Linear interpolation of observed pure-air markers, no model coefficients fit.

    It is a diagnostic control subtraction. Interpolation curvature and unknown
    experimental errors are not quantified, so no statistical bound is attached.
    """
    air=sorted((p['superficial_velocity_m_s'],p['gradient_pa_m']) for p in points if p['solids_mass_kg_s']==0)
    for (a,x),(b,y) in zip(air,air[1:]):
        if a<=U<=b:return x+(y-x)*(U-a)/(b-a)
    raise ValueError('Нет экспериментального контроля воздуха для интерполяции.')


def m3_narimatsu_benchmark(temperature_k=293.15,pressure_pa=101325.,roughness_m=.00009144,
                         include_digitization=True):
    evidence=m3_narimatsu_evidence();rows=[]
    options=dict(temperature_k=temperature_k,pressure_pa=pressure_pa,roughness_m=roughness_m)
    for p in evidence['points']:
        row=dict(p)
        try:
            row.update(status='computed',prediction=m3_narimatsu_local(p,**options))
            if p['solids_mass_kg_s'] and p['primary_curve_subset']:
                airp=dict(p,solids_mass_kg_s=0.)
                obsair=m3_interpolate_air(evidence['points'],p['superficial_velocity_m_s'])
                modelair=m3_narimatsu_local(airp,**options)['gradient_pa_m']
                row['air_control_subtraction']=dict(observed_air_interpolated_pa_m=obsair,
                    observed_particle_increment_pa_m=p['gradient_pa_m']-obsair,
                    predicted_particle_increment_pa_m=row['prediction']['gradient_pa_m']-modelair,
                    method='subtract interpolated observed air control and separately subtract model air prediction; no coefficients fitted',
                    interpolation_error_quantified=False)
            if include_digitization and p['primary_curve_subset']:
                endpoint_predictions=[];failures=[]
                for sign in (-1,1):
                    U=p['superficial_velocity_m_s']+sign*p['velocity_digitization_bound_m_s']
                    try:endpoint_predictions.append(m3_narimatsu_local(p,superficial_velocity=U,**options)['gradient_pa_m'])
                    except (ValueError,ArithmeticError) as exc:failures.append(str(exc))
                row['domain_stable_over_digitization']=not failures
                row['digitization_endpoint_failures']=failures
                if not failures:
                    lo=min(endpoint_predictions);hi=max(endpoint_predictions);obs=p['gradient_pa_m'];dy=p['gradient_digitization_bound_pa_m']
                    row['sampled_horizontal_digitization_prediction_range_pa_m']=[lo,hi]
                    row['compatible_with_digitization_box_only']=lo<=obs+dy and hi>=obs-dy
                    row['digitization_comparison_note']='Two U endpoints and vertical +/-2px only; unknown experimental measurement error is NOT included; not a confidence test or continuous enclosure.'
        except (ValueError,ArithmeticError) as exc:row.update(status='rejected',reason=str(exc))
        rows.append(row)
    air=[r for r in rows if r['solids_mass_kg_s']==0];solid=[r for r in rows if r['solids_mass_kg_s']>0]
    eligible=[r for r in solid if r['primary_curve_subset'] and r['status']=='computed']
    return dict(evidence=evidence,reference_assumptions=options,rows=rows,
        air_metrics=m3_gradient_metrics(air),solids_metrics=m3_gradient_metrics(solid),
        per_series={name:m3_gradient_metrics([r for r in solid if r['series_id']==name]) for name in sorted(set(r['series_id'] for r in solid))},
        eligible_solid_points=len(eligible),solid_points_stable_over_digitization=sum(r.get('domain_stable_over_digitization',False) for r in eligible),
        solid_points_compatible_with_digitization_only=sum(r.get('compatible_with_digitization_box_only',False) for r in eligible),
        status='conditional_experimental_curve_comparison_not_full_validation',coefficient_fitting=False,
        experimental_measurement_uncertainty_available=False,confidence_level=None,
        full_author_example_reproduced=False,validation_of_2mm_annulus=False,
        interpretation='Independent developed round-pipe data test the force closures conditionally. Good agreement of air alone does not validate particle closures or transfer to a narrow annulus.')


class Verification(Verification):
    def test_narimatsu_digitization_calibration_and_unique_observations(self):
        e=m3_narimatsu_evidence();p=e['points'];self.assertEqual(len(p),37)
        self.assertEqual(len({x['point_id'] for x in p}),37)
        self.assertEqual(sum(x['solids_mass_kg_s']==0 for x in p),15)
        a=p[0];self.assertAlmostEqual(a['superficial_velocity_m_s'],(121.38-44)*40/266)
        self.assertAlmostEqual(a['gradient_pa_m'],(165-148.08)*400/157)
        self.assertIsNone(a['measurement_error_bound_pa_m'])

    def test_developed_gradient_satisfies_solid_force_balance(self):
        p=next(x for x in m3_narimatsu_evidence()['points'] if x['series_id']=='glass_1mm_0041' and x['primary_curve_subset'])
        r=m3_narimatsu_local(p);f=r['forces_pa_m']
        self.assertLess(abs(f['drag_N_m3']-f['solids_wall_N_m3']-f['solids_weight_N_m3']),1e-9)
        self.assertLess(abs(r['equilibrium_dus_dz']),1e-10)
        self.assertFalse(r['yang_extrapolated'])

    def test_air_control_interpolation_exact_at_observed_marker(self):
        p=m3_narimatsu_evidence()['points'];a=p[7]
        self.assertAlmostEqual(m3_interpolate_air(p,a['superficial_velocity_m_s']),a['gradient_pa_m'])
        with self.assertRaises(ValueError):m3_interpolate_air(p,100.)

    def test_experimental_benchmark_does_not_claim_measurement_coverage(self):
        r=m3_narimatsu_benchmark();self.assertFalse(r['coefficient_fitting'])
        self.assertFalse(r['experimental_measurement_uncertainty_available'])
        self.assertFalse(r['validation_of_2mm_annulus']);self.assertIsNone(r['confidence_level'])
        self.assertEqual(r['air_metrics']['count'],15);self.assertEqual(r['solids_metrics']['count'],15)
        self.assertLess(r['solid_points_stable_over_digitization'],r['eligible_solid_points'])


class Verification(Verification):
    def test_existing_passport_export_menu_remains_accessible(self):
        import tempfile,io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.json';saved=Path(tmp)/'saved.json'
            extra=m3_example_passport(default_data())
            source.write_text(json.dumps(dict(schema_version=3,additional_inputs=asdict(extra))),encoding='utf-8')
            answers=iter(['6',str(source),'5','0'])
            with patch('builtins.input',side_effect=lambda _:next(answers)),patch.dict(globals(),{'unique_output':lambda *args:saved}),redirect_stdout(io.StringIO()):
                m3_full_menu(default_data(),None)
            restored=json.loads(saved.read_text(encoding='utf-8'))
            self.assertEqual(restored['schema_version'],3)
            self.assertEqual(restored['additional_inputs'],asdict(extra))


# Numerical quality tools. A refinement difference is not a physical error bound.
def m3_numerical_quality(data,flow,extra,levels=(2e-6,2e-7,2e-8),target_change_pa=.5,progress=False):
    if isinstance(extra,dict):extra=M3Passport(**extra)
    if len(levels)<2 or any(not math.isfinite(v) or not 0<v<=.01 for v in levels):
        raise ValueError('Нужны минимум два положительных допуска.')
    if any(b>=a for a,b in zip(levels,levels[1:])):raise ValueError('Допуски должны последовательно уменьшаться.')
    require_finite(target_change_pa)
    if target_change_pa<=0:raise ValueError('Положительный порог изменения обязателен.')
    rows=[]
    for i,rtol in enumerate(levels):
        if progress:print('Уточнение %d/%d, rtol=%.3g'%(i+1,len(levels),rtol))
        r=m3_full_circuit(data,flow,extra,rtol)
        row=dict(rtol=rtol,surface_supply_pressure_pa=r['surface_supply_pressure_pa'],
            return_bottom_pressure_pa=r['return_bottom_pressure_pa'],outlet_residual_pa=r['outlet_residual_pa'],
            pressure_boundary_tolerance_pa=r['numerical_tolerances']['pressure_boundary_pa'],
            return_steps=sum(p['steps'] for p in r['return_sections']),supply_steps=sum(p['steps'] for p in r['supply_sections']),
            gas_mass_relative_error=r['gas_mass_relative_error'],solids_mass_relative_error=r['solids_mass_relative_error'])
        if rows:
            row['supply_change_from_previous_pa']=row['surface_supply_pressure_pa']-rows[-1]['surface_supply_pressure_pa']
            row['bottom_change_from_previous_pa']=row['return_bottom_pressure_pa']-rows[-1]['return_bottom_pressure_pa']
        rows.append(row)
    return dict(model='M3 '+M3_VERSION,rows=rows,inputs=r['inputs'],additional_inputs=r['additional_inputs'],
        Q_reference_m3_min=r['Q_reference_m3_min'],target_change_pa=target_change_pa,
        observed_final_change_below_target=abs(rows[-1]['supply_change_from_previous_pa'])<=target_change_pa,
        physical_validation=False,confidence_level=None,certified_error_bound=False,
        note='Разность последних двух численных решений при неизменных уравнениях и входах. Это не оценка экспериментальной точности, не 95%-ный интервал и не доказанная верхняя граница ошибки.')


def m3_numerical_quality_menu(data,flow):
    print('Проверка численного уточнения М3: три решения с неизменными исходными данными.')
    print('1. Иллюстративный дополнительный паспорт\n2. Загрузить дополнительный паспорт JSON\n0. Назад')
    key=input('Пункт: ').strip()
    if key=='0':return
    if key=='1':extra=m3_example_passport(data)
    elif key=='2':
        payload=json.loads(Path(input('JSON-файл: ').strip().strip(chr(34))).read_text(encoding='utf-8-sig'))
        extra=M3Passport(**payload['additional_inputs'])
    else:raise ValueError('Выберите 0, 1 или 2.')
    print('Статус входов: '+extra.data_status)
    r=m3_numerical_quality(data,flow,extra,progress=True)
    print('rtol | Давление подачи, Па абс. | Изменение, Па | Невязка выхода, Па')
    for row in r['rows']:
        change=row.get('supply_change_from_previous_pa')
        print('%.3g | %.6f | %s | %.6g'%(row['rtol'],row['surface_supply_pressure_pa'],
            'первое решение' if change is None else '%+.6f'%change,row['outlet_residual_pa']))
    print(r['note'])
    path=unique_output(Path(__file__).resolve().parent,'numerical_quality','json')
    path.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Сохранено: '+str(path))


class Verification(Verification):
    def test_smooth_wall_diagnostic_and_controlled_models(self):
        diag=friction_regime_check(290.,.1,.001,.04,0.)
        f=diag['colebrook_f'];Re=diag['Re_gas']
        self.assertLess(abs(1/math.sqrt(f)+2*math.log10(2.51/(Re*math.sqrt(f)))),1e-10)
        self.assertIsNone(diag['nikuradse_f']);self.assertIsNone(diag['within_project_5pct_tolerance'])
        d=default_data();extra=replace(m3_example_passport(d),steel_roughness_m=0.,borehole_roughness_m=0.,core_roughness_m=0.)
        for model in ('M1_AIR','M2_shared_Chen'):
            r=m3_controlled_homogeneous(d,None,extra,model)
            self.assertGreater(r['surface_supply_pressure_pa'],0.)
        # Nikuradse's fully rough correlation still has no valid zero-roughness branch.
        with self.assertRaises(ValueError):m3_controlled_homogeneous(d,None,extra,'M2_Nikuradse')
        r=calculate(dict(d,guo_friction_mode=2),None,'guo');self.assertGreater(r['P4'],0.)

    def test_trace_solids_limit_has_finite_acceleration_and_continuous_pressure_gradient(self):
        c=m3_demo_case();gas_case=replace(c,solids_mass_kg_s=0.)
        reference=m3_derivative(gas_case,101325.,1.)[0];forces=[]
        for ms in (1e-12,1e-16,1e-20):
            tiny=replace(c,solids_mass_kg_s=ms);st=m3_local(tiny,101325.,1.);dp,dus=m3_derivative(tiny,101325.,1.)
            self.assertTrue(math.isfinite(dus));self.assertGreater(st['drag_N_m3'],0.)
            forces.append(st['drag_N_m3']/st['solids_volume_fraction'])
            self.assertLess(abs(dp-reference),1e-5)
        self.assertLess(max(forces)/min(forces)-1,1e-8)

    def test_numerical_jacobian_uses_inward_direction_near_yang_boundary(self):
        # ug/Ut is just above the domain limit; increasing p by the old
        # fixed Jacobian increment leaves it. The valid inward derivative exists.
        c=m3_demo_case();rho=101325/(c.gas_R*c.temperature_k)
        ut=m3_terminal_velocity(rho,c.particle_density_kg_m3,c.particle_diameter_m,c.viscosity_pa_s)
        alpha=c.solids_mass_kg_s/(c.area_m2*c.particle_density_kg_m3)
        mg=1.50000001*ut*c.area_m2*rho*(1-alpha)
        c=replace(c,gas_mass_kg_s=mg)
        st=m3_local(c,101325.,1.);self.assertGreater(st['gas_to_terminal_ratio'],1.5)
        stage=m3_implicit_stage(c,(101325.,1.),(101325.,1.),1e-6)
        self.assertTrue(all(math.isfinite(v) for v in stage))

    def test_refinement_quality_is_separate_from_physical_validation(self):
        d=dict(default_data(),dpth_h=3.,core_pipe_lngth_lk=.1)
        r=m3_numerical_quality(d,None,m3_example_passport(d),levels=(2e-6,2e-7))
        self.assertEqual(len(r['rows']),2);self.assertFalse(r['certified_error_bound'])
        self.assertFalse(r['physical_validation']);self.assertIsNone(r['confidence_level'])
        self.assertLess(abs(r['rows'][-1]['outlet_residual_pa']),2*r['rows'][-1]['pressure_boundary_tolerance_pa'])
        with self.assertRaises(ValueError):m3_numerical_quality(d,None,m3_example_passport(d),levels=(1e-6,1e-5))


class Verification(Verification):
    def test_shooting_secants_keep_bracket_and_check_true_residual(self):
        seen=[]
        def evaluate(p):
            self.assertTrue(1.<=p<=2.);seen.append(p)
            return dict(residual=p*p-2,reached_outlet=True)
        r=m3_shoot_root(evaluate,1.,2.,1e-9)
        self.assertLess(abs(r['pressure_pa']-math.sqrt(2)),1e-9)
        self.assertLess(r['bracket_width_pa'],1e-9)
        self.assertLess(abs(r['result']['residual']),1e-9)
        self.assertLess(len(seen),35)
        with self.assertRaises(M3DomainError):m3_shoot_root(evaluate,1.,1.1,1e-9)

if __name__=='__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print('\nРабота завершена.')
