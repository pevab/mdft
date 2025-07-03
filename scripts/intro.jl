using DrWatson
@quickactivate "mdft"

using LinearAlgebra
using Plots
using NNlib
using Distributions

#=
ATTENTION PROCESSES
=#

abstract type  AttentionProcess
end

struct BernoulliAttentionProcess <: AttentionProcess
    probs :: Vector{Float64}
end

function next(process :: BernoulliAttentionProcess) :: Vector{Float64}
    probs = process.probs
    n_attributes = length(probs)
    focus = rand(Categorical(probs))
    result = zeros(n_attributes)
    result[focus] = 1
    result
end

#=
MDFT STRUCTURE
=#

struct MDFT 
    n_alternatives :: Int64
    n_attributes :: Int64
    personal_evaluation_matrix :: Matrix{Float64}
    attention_process :: AttentionProcess
    contrast_matrix :: Matrix{Float64}
    residual_error_law :: Distribution
    feedback_matrix :: Matrix{Float64}
end

#=
STOPPING RULES
=#

abstract type StoppingRule end

struct ThresholdRule <: StoppingRule
    threshold :: Float64
end

function should_stop(rule :: ThresholdRule, history :: Matrix{Float64}) :: Bool
    (_, elapsed_time) = size(history)
    last_state = history[:, elapsed_time]
    threshold = rule.threshold
    return maximum(last_state) >= threshold
end

struct DeadlineRule <: StoppingRule
    deadline :: Int64
end

function should_stop(rule :: DeadlineRule, history :: Matrix{Float64}) :: Bool
    (_, elapsed_time) = size(history)
    deadline = rule.deadline
    return elapsed_time >= deadline
end

#=
SIMULATIONS
=#

function run(mdft :: MDFT, rule :: StoppingRule) :: Matrix{Float64}
    n_alternatives = mdft.n_alternatives
    history = zeros((n_alternatives, 1))
    previous = history[:, end]
    feedback_matrix = mdft.feedback_matrix
    contrast_matrix = mdft.contrast_matrix
    personal_evaluation_matrix = mdft.personal_evaluation_matrix
    while !should_stop(rule, history)
        # new value
        attention_weights = next(mdft.attention_process)
        residual = rand(mdft.residual_error_law, n_alternatives)
        current = feedback_matrix * previous + contrast_matrix * personal_evaluation_matrix * attention_weights + residual

        # push
        history = hcat(history, current)
        previous = current
    end
    history
end

#=
EXPERIMENT
=#

struct Measure
    model :: MDFT
    n_seeds :: Int
    max_time :: Int
end

struct MeasureResult
    mean :: Float64
    std :: Float64
end

function run_measure(measure :: Measure) :: Matrix{MeasureResult}
    function winner(p :: Vector{Float64}) :: Vector{Float64}
        probs = softmax(p)
        w = rand(Categorical(probs))
        r = zeros(size(p))
        r[w] = 1
        r
    end
    model = measure.model
    n_alternatives = model.n_alternatives
    max_time = measure.max_time
    n_seeds = measure.n_seeds
    results = Matrix{MeasureResult}(undef, n_alternatives, max_time)
    for deadline in 1:max_time
        s = zeros((n_alternatives,0))
        for _ in 1:n_seeds
            history = run(model, DeadlineRule(deadline))
            p = history[:, end]
            s = hcat(s, winner(p))
        end
        s_mean = mean(s, dims=2)
        s_std = std(s, dims=2)
        results[:, deadline] = [MeasureResult(s_mean[a], s_std[a]) for a in 1:n_alternatives]
    end
    results
end


#=
Examples
=#

function contrast_matrix(n_alternatives) :: Matrix{Float64}
    f = 1/(n_alternatives - 1)
    (1 + f) * Matrix(I, n_alternatives, n_alternatives) - f * ones((n_alternatives, n_alternatives))
end

function sym(m :: Matrix{Float64}) :: Matrix{Float64}
    (m + m')/2
end

function similarity_effect()
    max_time = 100
    n_seeds = 1000
    attention_process = BernoulliAttentionProcess([.49, .51])
    measure_ab = Measure(
        MDFT(
            2, # alternatives (cars) A B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(2),
            # residual error law
            Dirac(0),
            sym([ # feedback matrix S
                .940 .000;
                .000 .940;
            ])
        ),
        n_seeds,
        max_time,
    )
    measure_asb = Measure(
        MDFT(
            3, # alternatives (cars) A S B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                0.99  3.01; # S
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(3),
            # residual error law
            Dirac(0),
            sym([ # feedback matrix S
                .940 -0.025 -0.010;
                .000   .940 -0.010;
                .000   .000   .940;
            ])
        ),
        n_seeds,
        max_time,
    )

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    lo_ab = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_ab = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_ab = plot(mean_ab', ribbon=[lo_ab hi_ab], linecolor=[:blue :green], labels=['A' 'B'], ylim=(0,1))

    # ASB
    results_asb = run_measure(measure_asb)
    mean_asb = map(r -> r.mean, results_asb)
    lo_asb = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_asb = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_asb = plot(mean_asb', ribbon=[lo_asb hi_asb], linecolor=[:blue :red :green], labels=['A' 'S' 'B'], ylim=(0,1))

    # plot
    plot(plt_ab, plt_asb, layout=(1,2), title="Similarity effect")
end

function attraction_effect()
    max_time = 100
    n_seeds = 1000
    attention_process = BernoulliAttentionProcess([.51, .49])
    measure_ab = Measure(
        MDFT(
            2, # alternatives (cars) A B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(2),
            # residual error law
            Dirac(0),
            sym([ # feedback matrix S
                .940 .000;
                .000 .940;
            ])
        ),
        n_seeds,
        max_time,
    )
    measure_adb = Measure(
        MDFT(
            3, # alternatives (cars) A D B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                0.50  2.50; # D
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(3),
            # residual error law
            Dirac(0),
            # sym([ # feedback matrix S
            #     .940 -0.025 -0.010;
            #     .000   .940 -0.010;
            #     .000   .000   .940;
            # ])
            sym([ # feedback matrix S
                .940 -0.050 -0.000;
                .000   .940 -0.000;
                .000   .000   .940;
            ])
        ),
        n_seeds,
        max_time,
    )

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    lo_ab = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_ab = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_ab = plot(mean_ab', ribbon=[lo_ab hi_ab], linecolor=[:blue :green], labels=['A' 'B'], ylim=(0,1))

    # ADB
    results_adb = run_measure(measure_adb)
    mean_adb = map(r -> r.mean, results_adb)
    lo_adb = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_adb = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_adb = plot(mean_adb', ribbon=[lo_adb hi_adb], linecolor=[:blue :red :green], labels=['A' 'D' 'B'], ylim=(0,1))

    # plot
    plot(plt_ab, plt_adb, layout=(1,2))
end

function compromise_effect()
    max_time = 200
    n_seeds = 1000
    attention_process = BernoulliAttentionProcess([.50, .50])
    measure_ab = Measure(
        MDFT(
            2, # alternatives (cars) A B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(2),
            # residual error law
            Dirac(0),
            sym([ # feedback matrix S
                .940 .000;
                .000 .940;
            ])
        ),
        n_seeds,
        max_time,
    )
    measure_acb = Measure(
        MDFT(
            3, # alternatives (cars) A C B
            2, # attributes E (economy) and Q (quality)
            [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                2.00  2.00; # C
                3.00  1.00; # B
            ],
            # attention weight process
            attention_process,
            # contrast matrix
            contrast_matrix(3),
            # residual error law
            Dirac(0),
            # sym([ # feedback matrix S
            #     .940 -0.025 -0.010;
            #     .000   .940 -0.010;
            #     .000   .000   .940;
            # ])
            sym([ # feedback matrix S
                .940 -0.200 -0.000;
                .000   .940 -0.199;
                .000   .000   .940;
            ])
        ),
        n_seeds,
        max_time,
    )

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    lo_ab = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_ab = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_ab = plot(mean_ab', ribbon=[lo_ab hi_ab], linecolor=[:blue :green], labels=['A' 'B'], ylim=(0,1))

    # ACB
    results_acb = run_measure(measure_acb)
    mean_acb = map(r -> r.mean, results_acb)
    lo_acb = map(r -> r.mean - 0.67*r.std, results_ab)
    hi_acb = map(r -> r.mean + 0.67*r.std, results_ab)
    plt_acb = plot(mean_acb', ribbon=[lo_acb hi_acb], linecolor=[:blue :red :green], labels=['A' 'C' 'B'], ylim=(0,1))

    # plot
    plot(plt_ab, plt_acb, layout=(1,2))
end