module SimParamModule
using ..Utils
using IniFile

@enum SimulationType SW SC
function to_simulation_type(str::AbstractString)
    s = lowercase(str)
    if s in ("sw", "schulten-wolynes")
        return SW
    elseif s in ("sc", "semi-classical", "semiclassical")
        return SC
    else
        error("Unknown simulation_type \"$str\". Expected \"SW\" or \"SC\".")
    end
end

@enum StateType Singlet Triplet
function to_state_type(str::AbstractString)
    s = lowercase(str)
    if s in ("singlet", "s")
        return Singlet
    elseif s in ("triplet", "t")
        return Triplet
    else
        error("Unknown simulation_type \"$str\". Expected \"Singlet\" or \"Triplet\".")
    end
end

struct SimParams
    simulation_type :: SimulationType
    B               :: Vector{Float64}   # 4-field magnetic-field list
    simulation_time :: Float64           #
    N_samples       :: UInt64
    output_folder   :: String
    seed            :: Vector{Int}          # variable-length list
    initial_state   :: StateType
    dt              :: Float64
end

function read_simparams(cfg::IniFile.Inifile)
    sec = "simulation parameters"

    simulation_type = to_simulation_type(get(cfg, sec, "simulation_type", "SW"))
    B               = vecparse(Float64, get(cfg, sec, "B", ""))
    simulation_time = parse(Float64, get(cfg, sec, "simulation_time", "NaN"))
    N_samples       = parse(UInt64, get(cfg, sec, "N_samples", "0"))
    output_folder   = get(cfg, sec, "output_folder", "out")
    seed            = vecparse(Int, get(cfg, sec, "seed", ""))
    initial_state   = to_state_type(get(cfg, sec, "initial_state", "singlet"))
    dt              = parse(Float64, get(cfg, sec, "dt", "NaN"))
    @assert dt > 0
    @assert simulation_time > 0

    return SimParams(
        simulation_type,
        B,
        simulation_time,
        N_samples,
        output_folder,
        seed,
        initial_state,
        dt,
    )
end

export SimParams, read_simparams

end # module
