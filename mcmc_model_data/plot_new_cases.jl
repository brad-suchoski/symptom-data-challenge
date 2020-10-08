using Dates, Plots
function plot_new_cases(fips, model_file, hopkins_file)
    model = sort(filter(r->r.fips == fips,
                        CSV.File(model_file)|>DataFrame),
                 :Date)
    tmp = filter(r->r.FIPS !== missing && r.FIPS == fips, CSV.File(hopkins_file)|>DataFrame)
    reported = DataFrame(:Date => Date.(String.(names(tmp)[12:end]), "m/d/y") .+ Year(2000),
                         :Cases => collect(tmp[1,12:end]))
    plot(model.Date, model.NewI, label = "Model")
    plot!(reported.Date[2:end], diff(reported.Cases), label = "Reported")
    ylabel!("New Cases Per Day")
    title!(model.Jurisdiction[1])
end
