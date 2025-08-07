using IniFile, StaticArrays
using ..Utils


struct System
    J  ::Float64            # exchange coupling (scalar)
    D  ::SMatrix{3,3,Float64}  # 3×3 zero-trace tensor (StaticArrays keeps it fast)
    kS ::Float64            # singlet rate constant in μs-1
    kT ::Float64            # triplet  rate constant in μs-1
end

function read_system(cfg::IniFile.Inifile)
    sec = "system variables"

    J  = parse(Float64, get(cfg, sec, "J",  "NaN"))
    D  = mat3static(get(cfg,   sec, "D",  ""))
    kS = parse(Float64, get(cfg, sec, "kS", "NaN"))
    kT = parse(Float64, get(cfg, sec, "kT", "NaN"))

    return System(J, D, kS, kT)
end

export System, read_system
