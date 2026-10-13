from unittest import result
from inspect import Parameter
from re import A, match
import matplotlib.pyplot as plt
from tkinter import ROUND
from turtle import lt
import math
import subprocess
import os
import csv


def print_result(result):
    print("\nРезультаты расчёта")
    for key, value in result.items():
        print(f"{key}: {value}")


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


data = default_data()


def default_problem():
    return {
        "objective": {
            "parameter": "P4",
            "value": 0.67,
        },
        "mode": "target",
        "constraints": {
            "N_max": None,
            "P4_max": None,
            "vkol_min": None,
            "vbt_max": None}
    }


problem = default_problem()


def validate_problem(problem):
    required_keys = [
        "mode",
        "objective",
        "constraints"
    ]
    for key in required_keys:
        if key not in problem:
            raise ValueError(f"Нет ключа {key}")
    if problem["mode"] not in [
        "target",
        "max",
        "min"
    ]:
        raise ValueError(
            f"Неизвестный режим {problem['mode']}"
        )
    if "parameter" not in problem["objective"]:
        raise ValueError(
            "Не задан параметр цели"
        )
    return True


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


def calculate_state(data, state):
    result = {
        "inputs": {},
        "outputs": {},
        "checks": {}
    }
    result["inputs"] = state.copy()
    G = state["G"]
    hydraulic = hydraulic_model(
        data,
        G
    )
    result["outputs"] = hydraulic
    return result


def default_variables():
    return {
        "G": {
            "value": None,
            "active": True,
            "min": None,
            "max": None,
            "step": None,
            "direction": None
        },
        "N": {
            "value": None,
            "active": True,
            "min": None,
            "max": None,
            "step": None,
            "direction": None
        },
        "P4": {
            "value": None,
            "active": False,
            "min": None,
            "max": None,
            "step": None,
            "direction": None
        }
    }


variables = default_variables()
variables["G"]["value"] = 40
variables["G"]["min"] = 1
variables["G"]["max"] = 100
variables["G"]["active"] = True
variables["G"]["step"] = 0.2
variables["G"]["direction"] = 1
variables["N"]["value"] = None
variables["N"]["min"] = 0.1
variables["N"]["max"] = 50000.0
variables["N"]["active"] = False
variables["N"]["step"] = 10
variables["N"]["direction"] = 1


def validate_variables(variables, problem):
    errors = []
    objective_parameter = problem["objective"]["parameter"]
    for name, var in variables.items():
        required = [
            "value",
            "active",
            "min",
            "max",
            "step"
        ]
        for key in required:
            if key not in var:
                errors.append(
                    f"{name}: отсутствует ключ {key}"
                )
        if var["active"]:
            if var["value"] is None:
                errors.append(
                    f"{name}: нет начального значения"
                )
            if var["min"] is None:
                errors.append(
                    f"{name}: не задан минимум"
                )
            if var["max"] is None:
                errors.append(
                    f"{name}: не задан максимум"
                )
            if var["step"] is None:
                errors.append(
                    f"{name}: не задан шаг"
                )
            if var["direction"] not in (-1, 1):
                errors.append(
                    f"{name}: direction должен быть -1 или 1"
                )
        if (
                var["min"] is not None
                and var["max"] is not None
        ):
            if var["min"] > var["max"]:
                errors.append(
                    f"{name}: min больше max"
                )
        if var["step"] is not None:

            if var["step"] <= 0:
                errors.append(
                    f"{name}: шаг должен быть > 0"
                )
        if name == objective_parameter:

            if var["active"]:
                errors.append(
                    f"{name}: нельзя менять целевой параметр"
                )
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def apply_variables(variables):
    state = {}

    for name, var in variables.items():

        if var["value"] is not None:
            state[name] = var["value"]

    return state


def check_constraints(result, problem):
    checks = {
        "valid": True,
        "violations": []
    }
    outputs = result["outputs"]
    constraints = problem["constraints"]
    max_limits = {
        "N_max": "N",
        "P4_max": "P4",
        "vbt_max": "vbt"
    }
    min_limits = {
        "vkol_min": "vkol"
    }
    for limit, parameter in max_limits.items():
        value = constraints.get(limit)
        if value is not None:
            if outputs[parameter] > value:
                checks["valid"] = False
                checks["violations"].append(
                    f"{parameter} превышает {limit}"
                )
    for limit, parameter in min_limits.items():
        value = constraints.get(limit)
        if value is not None:
            if outputs[parameter] < value:
                checks["valid"] = False
                checks["violations"].append(
                    f"{parameter} меньше {limit}"
                )
    return checks


def evaluate_objective(result, problem):
    objective = problem["objective"]
    parameter = objective["parameter"]
    target = objective["value"]
    value = result["outputs"][parameter]
    error = value - target
    return {
        "parameter": parameter,
        "target": target,
        "value": value,
        "error": error,
        "abs_error": abs(error)
    }


def solver(data, problem, variables,
           tolerance=1e-6,
           max_iterations=100):
    # -----------------------------
    # Проверка постановки задачи
    # -----------------------------
    validate_problem(problem)
    check = validate_variables(
        variables,
        problem
    )
    if not check["valid"]:
        raise ValueError(
            "\n".join(check["errors"])
        )
    # -----------------------------
    # Поиск активной переменной
    # -----------------------------
    active = None
    for name, var in variables.items():
        if var["active"]:
            active = name
            break
    if active is None:
        raise ValueError(
            "Нет активной переменной."
        )
    direction = variables[active]["direction"]

    # -----------------------------
    # Внутренняя функция одного расчёта
    # -----------------------------
    def evaluate(value):
        variables[active]["value"] = value
        state = apply_variables(
            variables
        )
        result = calculate_state(
            data,
            state
        )
        constraints = check_constraints(
            result,
            problem
        )
        evaluation = evaluate_objective(
            result,
            problem
        )
        return {
            "result": result,
            "constraints": constraints,
            "evaluation": evaluation
        }

    # -----------------------------
    # Начальные границы поиска
    # -----------------------------
    left = variables[active]["min"]
    right = variables[active]["max"]
    # Проверяем левую границу
    left_step = evaluate(left)

    left_error = (
            left_step["evaluation"]["error"]
            * direction
    )
    # Проверяем правую границу
    right_step = evaluate(right)
    right_error = (
            right_step["evaluation"]["error"]
            * direction
    )
    # Для бисекции нужен разный знак ошибки
    if left_error * right_error > 0:
        raise RuntimeError(
            "На заданном диапазоне "
            "нет решения."
        )
    # -----------------------------
    # Основной цикл бисекции
    # -----------------------------
    for iteration in range(max_iterations):
        value = (left + right) / 2
        step = evaluate(value)
        evaluation = step["evaluation"]
        constraints = step["constraints"]
        # Проверяем достижение цели
        if (
                constraints["valid"]
                and
                evaluation["abs_error"] <= tolerance
        ):
            return {
                "result": step["result"],
                "evaluation": evaluation,
                "iterations": iteration + 1,
                "variable": {
                    "name": active,
                    "value": value
                }
            }
        # Определяем, какую половину оставить
        current_error = (
                evaluation["error"]
                * direction
        )
        if current_error > 0:
            right = value
        else:
            left = value
    # -----------------------------
    # Решение не найдено
    # -----------------------------
    raise RuntimeError(
        "Достигнуто максимальное "
        "количество итераций."
    )


solution = solver(data, problem, variables)
print(solution)


def input_brhl_diam_D():
    d = input("Введите диаметр скважины (м): ")
    try:
        d = float(d.replace(',', '.'))
        if d <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_brhl_diam_D()
        return d
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 12.5).\n")
        return input_brhl_diam_D()


def input_dpth_h():
    h = input("Введите глубину скважины (м): ")
    try:
        h = float(h.replace(',', '.'))
        if h <= 0:
            print("Глубина должна быть положительным числом.\n")
            return input_dpth_h()
        return h
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 300).\n")
        return input_dpth_h()


def input_drilling_angle_alfa():
    alfa = input("Введите угол наклона к горизонту (град): ")
    try:
        alfa = float(alfa.replace(',', '.'))
        if alfa <= 0 or alfa >= 180:
            print("Угол должен быть положительным числом и меньше 180.\n")
            return input_drilling_angle_alfa()
        return alfa
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 90).\n")
        return input_drilling_angle_alfa()


def input_out_diam_dril_pipe_d():
    d = input("Введите наружный диаметр бурильной трубы (м): ")
    try:
        d = float(d.replace(',', '.'))
        if d <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_out_diam_dril_pipe_d()
        return d
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.05).\n")
        return input_out_diam_dril_pipe_d()


def input_in_diam_dril_pipe_d1():
    d = input("Введите внутренний диаметр бурильной трубы (м): ")
    try:
        d = float(d.replace(',', '.'))
        if d <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_in_diam_dril_pipe_d1()
        return d
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.039).\n")
        return input_in_diam_dril_pipe_d1()


def input_cnctn_type_j():
    j = input("Выберите тип соединения бурильной трубы (1 - муфтово-замковый, 2 - нипельный, 3 - ССК): ")
    try:
        j = int(j)
        if j not in [1, 2, 3]:
            print("Некорректное значение. Введите 1, 2 или 3.\n")
            return input_cnctn_type_j()
        return j
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (1, 2 или 3).\n")
        return input_cnctn_type_j()


def input_diam_chnl_join_d3():
    d = input("Введите диаметр канала соединения (м): ")
    try:
        d = float(d.replace(',', '.'))
        if d <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_diam_chnl_join_d3()
        return d
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.028).\n")
        return input_diam_chnl_join_d3()


def input_pipe_lngth_lt():
    lt = input("Введите длину бурильной трубы (м): ")
    try:
        lt = float(lt.replace(',', '.'))
        if lt <= 0:
            print("Длина должна быть положительным числом.\n")
            return input_pipe_lngth_lt()
        return lt
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 4.5).\n")
        return input_pipe_lngth_lt()


def input_out_diam_core_pipe_dh():
    dh = input("Введите наружный диаметр колонковой трубы (м): ")
    try:
        dh = float(dh.replace(',', '.'))
        if dh <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_out_diam_core_pipe_dh()
        return dh
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.089).\n")
        return input_out_diam_core_pipe_dh()


def input_in_diam_core_pipe_db():
    db = input("Введите внутренний диаметр колонковой трубы (м): ")
    try:
        db = float(db.replace(',', '.'))
        if db <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_in_diam_core_pipe_db()
        return db
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.079).\n")
        return input_in_diam_core_pipe_db()


def input_core_pipe_lngth_lk():
    lk = input("Введите длину колонковой трубы (м): ")
    try:
        lk = float(lk.replace(',', '.'))
        if lk <= 0:
            print("Длина должна быть положительным числом.\n")
            return input_core_pipe_lngth_lk()
        return lk
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 5).\n")
        return input_core_pipe_lngth_lk()


def input_core_diam_dk():
    dk = input("Введите диаметр керна (м): ")
    try:
        dk = float(dk.replace(',', '.'))
        if dk <= 0:
            print("Диаметр должен быть положительным числом.\n")
            return input_core_diam_dk()
        return dk
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 0.073).\n")
        return input_core_diam_dk()


def input_core_out_bk():
    bk = input("Введите выход керна (м): ")
    try:
        bk = float(bk.replace(',', '.'))
        if bk <= 0 or bk >= 100:
            print("Выход керна должен быть положительным числом между 0 и 100.\n")
            return input_core_out_bk()
        return bk
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 80).\n")
        return input_core_out_bk()


def input_rock_density_ro():
    ro = input("Введите плотность породы (кг/м³): ")
    try:
        ro = float(ro.replace(',', '.'))
        if ro <= 0:
            print("Плотность должна быть положительным числом.\n")
            return input_rock_density_ro()
        return ro
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 2300).\n")
        return input_rock_density_ro()


def input_mech_spd_vm():
    vm = input("Введите механическую скорость (м/ч): ")
    try:
        vm = float(vm.replace(',', '.'))
        if vm <= 0:
            print("Скорость должна быть положительным числом.\n")
            return input_mech_spd_vm()
        return vm
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 6).\n")
        return input_mech_spd_vm()


def input_tool_type_j1():
    j1 = input("Выберите тип инструмента (1 - алмазная коронка, 2 - твёрдосплавная коронка, 3 - долото): ")
    try:
        j1 = int(j1)
        if j1 not in [1, 2, 3]:
            print("Некорректное значение. Введите 1, 2 или 3.\n")
            return input_tool_type_j1()
        return j1
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (1, 2 или 3).\n")
        return input_tool_type_j1()


def input_well_temp_t():
    t = input("Введите температуру в скважине (К): ")
    try:
        t = float(t.replace(',', '.'))
        if t <= 0:
            print("Температура должна быть положительным числом.\n")
            return input_well_temp_t()
        return t
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 290).\n")
        return input_well_temp_t()


def input_rqrd_up_spd_v1():
    v1 = input("Введите требуемую скорость подъема (м/ч): ")
    try:
        v1 = float(v1.replace(',', '.'))
        if v1 <= 0:
            print("Скорость должна быть положительным числом.\n")
            return input_rqrd_up_spd_v1()
        return v1
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 6).\n")
        return input_rqrd_up_spd_v1()


def input_comprsr_effency_ni():
    ni = input("Введите КПД компрессора (от 0 до 100): ")
    try:
        ni = float(ni.replace(',', '.'))
        if ni <= 0 or ni > 100:
            print("КПД должно быть числом между 0 и 100.\n")
            return input_comprsr_effency_ni()
        return ni / 100
    except ValueError:
        print("Вы ввели некорректное значение. Введите число (например, 30).\n")
        return input_comprsr_effency_ni()


def metod_rash_kudr():
    while True:
        def metod_rash_kudr1():
            print("\nРежим ввода вручную")
            print("\nПараметры скважины")
            brhl_diam_D = input_brhl_diam_D()
            dpth_h = input_dpth_h()
            drilling_angle_alfa = input_drilling_angle_alfa()
            print("\nПараметры бурильной трубы")
            out_diam_dril_pipe_d = input_out_diam_dril_pipe_d()
            in_diam_dril_pipe_d1 = input_in_diam_dril_pipe_d1()
            cnctn_type_j = input_cnctn_type_j()
            diam_chnl_join_d3 = input_diam_chnl_join_d3()
            pipe_lngth_lt = input_pipe_lngth_lt()
            print("\nПараметры колонковой трубы")
            out_diam_core_pipe_dh = input_out_diam_core_pipe_dh()
            in_diam_core_pipe_db = input_in_diam_core_pipe_db()
            core_pipe_lngth_lk = input_core_pipe_lngth_lk()
            core_diam_dk = input_core_diam_dk()
            core_out_bk = input_core_out_bk()
            print("\nПараметры бурения")
            rock_density_ro = input_rock_density_ro()
            mech_spd_vm = input_mech_spd_vm()
            tool_type_j1 = input_tool_type_j1()
            print("\nПараметры воздуха и компрессора")
            well_temp_t = input_well_temp_t()
            rqrd_up_spd_v1 = input_rqrd_up_spd_v1()
            comprsr_effency_ni = input_comprsr_effency_ni()
            gast_koef_kr = 1.2
            coef_spd_unvnss_k = 1.2
            coef_red_lift_frc_k1 = 1.1
            ATM_PRSSR_P0 = 100000
            GAS_CONST_R = 287
            ACCLRTN_GRAV_g = 9.81
            AIR_ADBTC_INDX_k = 1.4
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
            air_cons_q = 47.1 * coef_spd_unvnss_k * coef_red_lift_frc_k1 * (
                        brhl_diam_D ** 2 - out_diam_dril_pipe_d ** 2) * rqrd_up_spd_v1
            air_density_ron = ATM_PRSSR_P0 / (GAS_CONST_R * 273)
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
                'Восходящее давление в кольцевом пространстве бурильной трубы': round(P1, 3),
                'Восходящее давление в кольцевом пространстве колонковой трубы': round(P2, 3),
                'Нисходящее давление в кольцевом пространстве колонковой трубы': round(P3, 3),
                'Нисходящее давление во внутреннем канале бурильной трубы': round(P4, 3),
                'Общее давление': round(Pres, 3),
                'Мощность компрессора': round(N),
                'Скорость потока воздуха в кольцевом пространстве': round(vkol, 2),
                'Скорость потока воздуха в бурильной трубе': round(vbt, 2),
                'Массовый расход воздуха': round(mass_air_cons_g, 3),
                'Объемный расход воздуха': round(air_cons_q, 3)}
            return 0

        def metod_rash_kudr2():
            print("\nРежим автоматического ввода")
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
            air_cons_q = 47.1 * coef_spd_unvnss_k * coef_red_lift_frc_k1 * (
                        brhl_diam_D ** 2 - out_diam_dril_pipe_d ** 2) * rqrd_up_spd_v1
            air_density_ron = ATM_PRSSR_P0 / (GAS_CONST_R * 273)
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
                'Восходящее давление в кольцевом пространстве бурильной трубы': round(P1, 3),
                'Восходящее давление в кольцевом пространстве колонковой трубы': round(P2, 3),
                'Нисходящее давление в кольцевом пространстве колонковой трубы': round(P3, 3),
                'Нисходящее давление во внутреннем канале бурильной трубы': round(P4, 3),
                'Общее давление': round(Pres, 3),
                'Мощность компрессора': round(N),
                'Скорость потока воздуха в кольцевом пространстве': round(vkol, 2),
                'Скорость потока воздуха в бурильной трубе': round(vbt, 2),
                'Массовый расход воздуха': round(mass_air_cons_g, 3),
                'Объемный расход воздуха': round(air_cons_q, 3)}

        print("\nМетод расчёта величин по И.О.Кудряшову")
        print("Ввод параметров будет происходить вручную?\n1 - Да\n2 - Нет")
        vibor_vvod_kudr = input("Ваш выбор: ")
        if not vibor_vvod_kudr.isdigit():
            print("Вы ввели некорректное значение. Введите число.\n")
            return metod_rash_kudr()
        elif int(vibor_vvod_kudr) == 1:
            result = metod_rash_kudr1()
            break
        elif int(vibor_vvod_kudr) == 2:
            result = metod_rash_kudr2()
            break
        else:
            print("Вы ввели некорректное значение. Попробуйте снова.\n")
            return metod_rash_kudr()
            continue
    print_result(result)
    return 0

    print_result(result)


# def hydraulic_model(data, G):
#     brhl_diam_D = data["brhl_diam_D"]
#     dpth_h = data["dpth_h"]
#     drilling_angle_alfa = data["drilling_angle_alfa"]
#     out_diam_dril_pipe_d = data["out_diam_dril_pipe_d"]
#     in_diam_dril_pipe_d1 = data["in_diam_dril_pipe_d1"]
#     cnctn_type_j = data["cnctn_type_j"]
#     diam_chnl_join_d3 = data["diam_chnl_join_d3"]
#     pipe_lngth_lt = data["pipe_lngth_lt"]
#     out_diam_core_pipe_dh = data["out_diam_core_pipe_dh"]
#     in_diam_core_pipe_db = data["in_diam_core_pipe_db"]
#     core_pipe_lngth_lk = data["core_pipe_lngth_lk"]
#     core_diam_dk = data["core_diam_dk"]
#     core_out_bk = data["core_out_bk"]
#     rock_density_ro = data["rock_density_ro"]
#     mech_spd_vm = data["mech_spd_vm"]
#     tool_type_j1 = data["tool_type_j1"]
#     gast_koef_kr = data["gast_koef_kr"]
#     coef_spd_unvnss_k = data["coef_spd_unvnss_k"]
#     coef_red_lift_frc_k1 = data["coef_red_lift_frc_k1"]
#     well_temp_t = data["well_temp_t"]
#     rqrd_up_spd_v1 = data["rqrd_up_spd_v1"]
#     comprsr_effency_ni = data["comprsr_effency_ni"]
#     ATM_PRSSR_P0 = data["ATM_PRSSR_P0"]
#     GAS_CONST_R = data["GAS_CONST_R"]
#     ACCLRTN_GRAV_g = data["ACCLRTN_GRAV_g"]
#     AIR_ADBTC_INDX_k = data["AIR_ADBTC_INDX_k"]
#     drill_pipe_lngth_hbt = dpth_h - core_pipe_lngth_lk
#     a0 = calc_koef_a0(cnctn_type_j)
#     f1 = math.pi * (brhl_diam_D ** 2 - out_diam_dril_pipe_d ** 2) / 4
#     f2 = math.pi * in_diam_dril_pipe_d1 ** 2 / 4
#     f3 = math.pi * (brhl_diam_D ** 2 - out_diam_core_pipe_dh ** 2) / 4
#     f4 = math.pi * (in_diam_core_pipe_db ** 2 - core_diam_dk ** 2) / 4
#     De1 = brhl_diam_D - out_diam_dril_pipe_d
#     De2 = in_diam_dril_pipe_d1
#     De3 = brhl_diam_D - out_diam_core_pipe_dh
#     De4 = in_diam_core_pipe_db - core_diam_dk
#     air_density_ron = ATM_PRSSR_P0 / (GAS_CONST_R * 273)
#     mass_air_cons_g = G
#     air_cons_q = mass_air_cons_g * 60 / air_density_ron
#     mass_rock_face_gz = math.pi / 4 * (brhl_diam_D ** 2 - core_out_bk * core_diam_dk ** 2 / 100) * rock_density_ro * mech_spd_vm / 3600
#     sludge_conc_nu = mass_rock_face_gz / mass_air_cons_g
#     c = GAS_CONST_R * well_temp_t
#     b = ACCLRTN_GRAV_g * math.sin(drilling_angle_alfa * math.pi / 180) / (2 * c)
#     e = mass_air_cons_g ** 2 * c
#     a1 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De1, f1)
#     a2 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De2, f2)
#     a3 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De3, f3)
#     a4 = calc_an(mass_air_cons_g, GAS_CONST_R, well_temp_t, De4, f4)
#     lambd1 = calc_lambdn(De1)
#     lambd2 = calc_lambdn(De2)
#     lambd3 = calc_lambdn(De3)
#     lambd4 = calc_lambdn(De4)
#     nc = math.ceil(drill_pipe_lngth_hbt / pipe_lngth_lt)
#     csi = a0 * ((in_diam_dril_pipe_d1 / diam_chnl_join_d3) ** 2 - 1) ** 2
#     le = in_diam_dril_pipe_d1 * nc * csi / lambd2
#     P1 = calc_pressure_up(ATM_PRSSR_P0, lambd1, gast_koef_kr, sludge_conc_nu, a1, drill_pipe_lngth_hbt, b)
#     P2 = calc_pressure_up(P1 * 10 ** 6, lambd3, gast_koef_kr, sludge_conc_nu, a3, core_pipe_lngth_lk, b)
#     P3 = calc_pressure_down(P2 * 10 ** 6, lambd4, a4, core_pipe_lngth_lk, 0, b)
#     P4 = calc_pressure_down(P3 * 10 ** 6, lambd2, a2, drill_pipe_lngth_hbt, le, b)
#     Pres = P4 + 0.05
#     N = AIR_ADBTC_INDX_k / (AIR_ADBTC_INDX_k - 1) * ATM_PRSSR_P0 * air_cons_q / 60 * ((Pres * 10 ** 6 / ATM_PRSSR_P0) ** ((coef_spd_unvnss_k - 1) / coef_spd_unvnss_k) - 1) / comprsr_effency_ni
#     vkol = air_cons_q / (60 * f1)
#     vbt = air_cons_q / (60 * f2)
#     return {
#         "P1": P1,
#         "P2": P2,
#         "P3": P3,
#         "P4": P4,
#         "Pres": Pres,
#         "N": N,
#         "vkol": vkol,
#         "vbt": vbt
#     }
#     print_result(result)
def check_constraints(result,
                      N_max=40000,
                      Vkol_min=12,
                      Vbt_max=120,
                      P4_max=5):
    print("\n" + "=" * 55)
    print("ПРОВЕРКА ДОПУСТИМОСТИ РЕЖИМА")
    print("=" * 55)
    ok = True
    if result["vkol"] >= Vkol_min:
        print(f"[ OK ] Скорость в кольцевом пространстве "
              f"{result['vkol']:.2f} м/с "
              f"(норма ≥ {Vkol_min})")
    else:
        print(f"[FAIL] Скорость в кольцевом пространстве "
              f"{result['vkol']:.2f} м/с "
              f"(требуется ≥ {Vkol_min})")
        ok = False
    if result["vbt"] <= Vbt_max:
        print(f"[ OK ] Скорость в бурильной трубе "
              f"{result['vbt']:.2f} м/с "
              f"(норма ≤ {Vbt_max})")
    else:
        print(f"[FAIL] Скорость в бурильной трубе "
              f"{result['vbt']:.2f} м/с "
              f"(допустимо ≤ {Vbt_max})")
        ok = False
    if result["P4"] <= P4_max:
        print(f"[ OK ] Давление P4 "
              f"{result['P4']:.3f} МПа "
              f"(норма ≤ {P4_max})")
    else:
        print(f"[FAIL] Давление P4 "
              f"{result['P4']:.3f} МПа "
              f"(допустимо ≤ {P4_max})")
        ok = False
    if result["N"] <= N_max:
        print(f"[ OK ] Мощность компрессора "
              f"{result['N']:.1f} Вт "
              f"(норма ≤ {N_max:.1f})")
    else:
        print(f"[FAIL] Мощность компрессора "
              f"{result['N']:.1f} Вт "
              f"(допустимо ≤ {N_max:.1f})")
        ok = False
    print("=" * 55)
    if ok:
        print("ИТОГ: РЕЖИМ ДОПУСТИМ")
    else:
        print("ИТОГ: РЕЖИМ НЕ ДОПУСТИМ")
    print("=" * 55)
    return ok


def get_parameter(G, parameter):
    data = default_data()
    result = hydraulic_model(data, G)
    if parameter not in result:
        print(f"Параметр '{parameter}' отсутствует в модели.")
    return result[parameter]


def residual(G, parameter, target_value):
    value = get_parameter(G, parameter)
    return value - target_value


def check_target(parameter, target, Gmin=0.01, Gmax=2):
    names = {
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "P4",
        "Pres": "Pres",
        "N": "N",
        "vkol": "Vкольц",
        "vbt": "VБТ",
        "nu": "ν"
    }
    symbol = names.get(parameter, parameter)
    f1 = residual(Gmin, parameter, target)
    f2 = residual(Gmax, parameter, target)
    value1 = get_parameter(Gmin, parameter)
    value2 = get_parameter(Gmax, parameter)
    print(f"{symbol}({Gmin:.3f}) = {value1:.4f}")
    print(f"{symbol}({Gmax:.3f}) = {value2:.4f}")
    print(f"Целевое значение = {target:.4f}")
    if f1 * f2 > 0:
        print("\nКорень не гарантирован на выбранном интервале.")
        return False
    print("\nКорень находится внутри интервала.")
    return True


def bisection(parameter,
              target,
              Gmin=0.01,
              Gmax=100,
              eps=1e-5,
              max_iter=100):
    if not check_target(parameter, target, Gmin, Gmax):
        return None
    history = []
    print("\nМетод бисекции\n")
    print(f"{'Итер':>4} {'G':>10} {'Параметр':>12} {'Невязка':>12}")
    for i in range(1, max_iter + 1):
        Gmid = (Gmin + Gmax) / 2
        value = get_parameter(Gmid, parameter)
        fmid = residual(Gmid, parameter, target)
        history.append({
            "iter": i,
            "Gmin": Gmin,
            "Gmax": Gmax,
            "Gmid": Gmid,
            "value": value,
            "residual": fmid
        })
        print(f"{i:4d} {Gmid:10.6f} {value:12.6f} {fmid:12.6f}")
        if abs(fmid) < eps:
            print("\nРешение найдено.")
            return {
                "G": Gmid,
                "value": value,
                "iterations": i,
                "history": history
            }
        fmin = residual(Gmin, parameter, target)
        if fmin * fmid < 0:
            Gmax = Gmid
        else:
            Gmin = Gmid
    print("\nДостигнут предел итераций.")
    return None


def input_parameter():
    parameter = input("Введите параметр для расчёта (P1, P2, P3, P4, Pres, N, vkol, vbt): ")
    if parameter not in ["P1", "P2", "P3", "P4", "Pres", "N", "vkol", "vbt"]:
        print("Некорректный параметр. Пожалуйста, выберите из списка: P1, P2, P3, P4, Pres, N, vkol, vbt.")
        return input_parameter()
    return parameter


parameter = input_parameter()
result = bisection(parameter, target=0.67)
if result is not None:
    data = default_data()
    print("\nИТОГ")
    print(f"G = {result['G']:.6f} кг/с")
    print(f"{parameter} = {result['value']:.6f}")
    print(f"Итераций = {result['iterations']}")
    model = hydraulic_model(data, result["G"])
    check_constraints(model)


def scan_G():
    G = 0.05
    step = 0.05
    G_max = 100.00
    print(f"{'G':>6} {'P4':>8} {'N':>12} {'vkol':>8} {'dP4/dG':>10} {'dN/dG':>12} {'dv/dG':>10}")
    print("-" * 85)
    G_values = []
    P1_values = []
    P2_values = []
    P3_values = []
    P4_values = []
    N_values = []
    Vkol_values = []
    Vbt_values = []
    dP4_values = []
    dN_values = []
    dV_values = []
    prev_P4 = None
    prev_N = None
    prev_V = None
    while G <= G_max:
        result = hydraulic_model(G)
        P1 = result["P1"]
        P2 = result["P2"]
        P3 = result["P3"]
        P4 = result["P4"]
        N = result["N"]
        vkol = result["vkol"]
        vbt = result["vbt"]
        G_values.append(G)
        P1_values.append(P1)
        P2_values.append(P2)
        P3_values.append(P3)
        P4_values.append(P4)
        N_values.append(N)
        Vkol_values.append(vkol)
        Vbt_values.append(vbt)
        if prev_P4 is None:
            dP4 = ""
            dN = ""
            dV = ""
            print(f"{G:6.2f} {P4:8.3f} {N:12.1f} {vkol:8.2f} {'-':>10} {'-':>12} {'-':>10}")
        else:
            dP4 = (P4 - prev_P4) / step
            dN = (N - prev_N) / step
            dV = (vkol - prev_V) / step
            print(
                f"{G:6.2f} "
                f"{P4:8.3f} "
                f"{N:12.1f} "
                f"{vkol:8.2f} "
                f"{dP4:10.3f} "
                f"{dN:12.1f} "
                f"{dV:10.3f}"
            )
        dP4_values.append(dP4)
        dN_values.append(dN)
        dV_values.append(dV)
        prev_P4 = P4
        prev_N = N
        prev_V = vkol
        G += step
    with open("scan_G_results.csv",
              "w",
              newline="",
              encoding="utf-8-sig") as file:

        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "G, кг/с",
            "P1, МПа",
            "P2, МПа",
            "P3, МПа",
            "P4, МПа",
            "N, Вт",
            "vкольц, м/с",
            "vБТ, м/с",
            "dP4/dG",
            "dN/dG",
            "dv/dG"
        ])

        for i in range(len(G_values)):
            writer.writerow([
                G_values[i],
                P1_values[i],
                P2_values[i],
                P3_values[i],
                P4_values[i],
                N_values[i],
                Vkol_values[i],
                Vbt_values[i],
                dP4_values[i],
                dN_values[i],
                dV_values[i]
            ])

    print("\nCSV-файл scan_G_results.csv успешно сохранён.")


def programm_menu():
    subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)
    print("Меню программы расчёта показателей бурения с использованием воздуха\n")
    print("Выберите пункт программы, нажав соответствующую цифру на клавиатуре")
    print("1 - Расчёт по классическому методу И.О.Кудряшова")
    print("2 - Оптимизационный расчёт")
    vibor_podprogrami = input("Ваш выбор: ")
    if int(vibor_podprogrami) == 1:
        metod_rash_kudr()
    elif int(vibor_podprogrami) == 2:
        pass
    return 0
# programm_menu()