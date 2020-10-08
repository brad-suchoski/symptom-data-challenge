using Dates, DifferentialEquations, Interpolations, Plots

interp1d(x, y) = extrapolate(interpolate((x,), y, Gridded(Linear())), Flat())

struct Parameters
    βp
    t0
    βf
    σ
    γ
end

function seir(du, u, p, t)
    β = nothing == p.t0 || t <= p.t0 ? p.βp(t) : p.βf(t)
    N = sum(u)
    S2E = β * u[1] * u[3] / N
    E2I = p.σ * u[2]
    I2R = p.γ * u[3]
    copy!(du, [-S2E, S2E - E2I, E2I - I2R, I2R, E2I])
    return
end

function run_seir(fips, model_file, hopkins_file)
    model_data = sort(filter(r->r.fips == fips,
                             CSV.File(model_file)|>DataFrame),
                      :Date)
    tmp = filter(r->r.FIPS !== missing && r.FIPS == fips, CSV.File(hopkins_file)|>DataFrame)
    hopkins_data = DataFrame(:Date => Date.(String.(names(tmp)[12:end]), "m/d/y") .+ Year(2000),
                             :Cases => collect(tmp[1,12:end]))

    allts = [round(d - model_data.Date[1], Day).value for d in model_data.Date]
    u0 = Vector{Float64}([model_data[1,[:CurS, :CurE, :CurI, :CurR]]...; 0.0])
    βp = Vector(model_data.R_e ./ 5.2)
    params = Parameters(interp1d(allts, βp), nothing, nothing,
                        1. / 5.2, 1. / 5.2)
    prob = ODEProblem(seir, u0, (0., maximum(allts)), params)
    base_sol = solve(prob, Euler(); dt = .1)
    return base_sol, hopkins_data
end
