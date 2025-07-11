using DrWatson
@quickactivate "mdft"

using LinearAlgebra
using Plots
using NNlib
using Distributions

#= -- MDFT
We focus on 2 attributes each.
=#

#=
ATTENTION PROCESSES
=#

struct AttentionProcess
    p :: Float64
end

function next(process :: AttentionProcess) :: Vector{Float64}
    p = process.p
    we = rand(Bernoulli(p))
    wq = rand(Bernoulli(1 - p))
    [we, wq]
end

#=
MDFT STRUCTURE
=#


struct MDFT 
    n_alternatives :: Int
    phi1 :: Float64
    phi2 :: Float64
    beta :: Float64
    evaluation_matrix :: Matrix{Float64} # alternatives x attributes = alternatives x {economy, quality}
    attention_process :: AttentionProcess
    residual_error_law :: Distribution
end

function psy_dist2(mdft :: MDFT, a :: Int, b :: Int )
    m = mdft.evaluation_matrix
    de = m[a, 1] - m[b, 1]
    dq = m[a, 2] - m[b, 2]
    di = (de - dq)/sqrt(2)
    dd = (de + dq)/sqrt(2)
    di^2 + mdft.beta * dd^2
end

function contrast_matrix(mdft :: MDFT) :: Matrix{Float64}
    n = mdft.n_alternatives
    f = 1 / (n - 1)
    (1 + f) * Matrix(I, n, n) - f * ones((n, n))
end

function feedback_matrix(mdft :: MDFT) :: Matrix{Float64}
    n = mdft.n_alternatives
    phi1 = mdft.phi1
    phi2 = mdft.phi2
    F = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i != j
                F[i,j] = phi2 * exp(phi1 * psy_dist2(mdft, i, j))
            end
        end
    end
    Matrix(I, n, n) - F
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
    S = feedback_matrix(mdft)
    C = contrast_matrix(mdft)
    M = mdft.evaluation_matrix
    while !should_stop(rule, history)
        # new value
        W = next(mdft.attention_process)
        residual = rand(mdft.residual_error_law, n_alternatives)
        current = S * previous + C * M * W + residual

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
        _, w = findmax(p)
        # probs = softmax(p)
        # w = rand(Categorical(probs))
        
        # one-hot encoding
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

#=
SIMILARITY EFFECT

Due to
- more attention on the attribute Q (more favorable to A)
- contrast matrix
No lateral inhibition.
=#

function similarity_effect()
    max_time = 100
    n_seeds = 1000
    phi1 = 0.0
    phi2 = 0.1
    beta = 2.0
    attention_process = AttentionProcess(.49)
    error = Normal(0, 0.01)
    M = [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                0.99  3.01; # S
                3.00  1.00; # B
            ]
    measure_ab = Measure(
        MDFT(
            2,
            phi1,
            phi2,
            beta,
            M[[1,3],:],
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )
    measure_asb = Measure(
        MDFT(
            3,
            phi1,
            phi2,
            beta,
            M,
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )

    # Alternatives scatter plot
    scatter_asb = plot()
    scatter!(M[:,1]', M[:, 2]', title="Alternatives", 
        legend=true, 
        label=["A" "S" "B"], 
        color=[:blue :red :green], 
        xlim=(0,3.5), ylim=(0, 3.5),
        alpha=0.5)

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    std_ab = map(r -> r.std, results_ab)
    plt_ab = plot(title="Control")
    colors_ab = [:blue :green]
    labels_ab = ['A' 'B']
    for i in 1:2
        plot!(mean_ab[i, :], ribbon=0.67 * std_ab[i, :], linecolor=colors_ab[i], label=labels_ab[i], ylim=(0,1))
    end

    # ASB
    results_asb = run_measure(measure_asb)
    mean_asb = map(r -> r.mean, results_asb)
    std_asb = map(r -> r.std, results_asb)
    plt_asb = plot(title="Similarity")
    colors_asb = [:blue :red :green]
    labels_asb = ['A' 'S' 'B']
    for i in 1:3
        plot!(mean_asb[i, :], ribbon=0.67 * std_asb[i, :], linecolor=colors_asb[i], label=labels_asb[i], ylim=(0,1))
    end

    # plot
    plot(scatter_asb, plt_ab, plt_asb, layout=(1,3))
end

#=
ATTRACTION EFFECT

Due to
- lateral inhibition
- emphasis on the dominance direction (D) in the (I,D) space
=#

function attraction_effect()
    max_time = 100
    n_seeds = 1000
    phi1 = 0.1
    phi2 = 0.1
    beta = 10.0
    attention_process = AttentionProcess(.6) # advantage for E (more favorable for B)
    error = Normal(0, 0.01)
    M = [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                0.50  0.50; # D
                3.00  1.00; # B
            ]
    measure_ab = Measure(
        MDFT(
            2,
            phi1,
            phi2,
            beta,
            M[[1,3],:],
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )
    measure_asb = Measure(
        MDFT(
            3,
            phi1,
            phi2,
            beta,
            M,
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )

    # Alternatives scatter plot
    scatter_asb = plot()
    scatter!(M[:,1]', M[:, 2]', title="Alternatives", 
        legend=true, 
        label=["A" "D" "B"], 
        color=[:blue :red :green], 
        xlim=(0,3.5), ylim=(0, 3.5),
        alpha=0.5)

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    std_ab = map(r -> r.std, results_ab)
    plt_ab = plot(title="Control")
    colors_ab = [:blue :green]
    labels_ab = ['A' 'B']
    for i in 1:2
        plot!(mean_ab[i, :], ribbon=0.67 * std_ab[i, :], linecolor=colors_ab[i], label=labels_ab[i], ylim=(0,1))
    end

    # ASB
    results_asb = run_measure(measure_asb)
    mean_asb = map(r -> r.mean, results_asb)
    std_asb = map(r -> r.std, results_asb)
    plt_asb = plot(title="Attraction")
    colors_asb = [:blue :red :green]
    labels_asb = ['A' 'D' 'B']
    for i in 1:3
        plot!(mean_asb[i, :], ribbon=0.67 * std_asb[i, :], linecolor=colors_asb[i], label=labels_asb[i], ylim=(0,1))
    end

    # plot
    plot(scatter_asb, plt_ab, plt_asb, layout=(1,3))
end

#=
COMPROMISE EFFECT

Due to
=#

function compromise_effect()
    max_time = 100
    n_seeds = 1000
    phi1 = 0.0
    phi2 = 0.1
    beta = 10.0
    attention_process = AttentionProcess(.5) # fair process
    error = Normal(0, 0.01)
    M = [ # personal evaluation matrix M
                # E     Q
                1.00  3.00; # A
                2.00  2.00; # C
                3.00  1.00; # B
            ]
    measure_ab = Measure(
        MDFT(
            2,
            phi1,
            phi2,
            beta,
            M[[1,3],:],
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )
    measure_asb = Measure(
        MDFT(
            3,
            phi1,
            phi2,
            beta,
            M,
            attention_process,
            error,
        ),
        n_seeds,
        max_time,
    )

    # Alternatives scatter plot
    scatter_asb = plot()
    scatter!(M[:,1]', M[:, 2]', title="Alternatives", 
        legend=true, 
        label=["A" "C" "B"], 
        color=[:blue :red :green], 
        xlim=(0,3.5), ylim=(0, 3.5),
        alpha=0.5)

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    std_ab = map(r -> r.std, results_ab)
    plt_ab = plot(title="Control")
    colors_ab = [:blue :green]
    labels_ab = ['A' 'B']
    for i in 1:2
        plot!(mean_ab[i, :], ribbon=0.67 * std_ab[i, :], linecolor=colors_ab[i], label=labels_ab[i], ylim=(0,1))
    end

    # ASB
    results_asb = run_measure(measure_asb)
    mean_asb = map(r -> r.mean, results_asb)
    std_asb = map(r -> r.std, results_asb)
    plt_asb = plot(title="Compromise")
    colors_asb = [:blue :red :green]
    labels_asb = ['A' 'C' 'B']
    for i in 1:3
        plot!(mean_asb[i, :], ribbon=0.67 * std_asb[i, :], linecolor=colors_asb[i], label=labels_asb[i], ylim=(0,1))
    end

    # plot
    plot(scatter_asb, plt_ab, plt_asb, layout=(1,3))
end



# function compromise_effect()
#     max_time = 200
#     n_seeds = 1000
#     attention_process = AttentionProcess([.50, .50])
#     measure_ab = Measure(
#         MDFT(
#             2, # alternatives (cars) A B
#             2, # attributes E (economy) and Q (quality)
#             [ # personal evaluation matrix M
#                 # E     Q
#                 1.00  3.00; # A
#                 3.00  1.00; # B
#             ],
#             # attention weight process
#             attention_process,
#             # contrast matrix
#             contrast_matrix(2),
#             # residual error law
#             Dirac(0),
#             sym([ # feedback matrix S
#                 .940 .000;
#                 .000 .940;
#             ])
#         ),
#         n_seeds,
#         max_time,
#     )
#     measure_acb = Measure(
#         MDFT(
#             3, # alternatives (cars) A C B
#             2, # attributes E (economy) and Q (quality)
#             [ # personal evaluation matrix M
#                 # E     Q
#                 1.00  3.00; # A
#                 2.00  2.00; # C
#                 3.00  1.00; # B
#             ],
#             # attention weight process
#             attention_process,
#             # contrast matrix
#             contrast_matrix(3),
#             # residual error law
#             Dirac(0),
#             # sym([ # feedback matrix S
#             #     .940 -0.025 -0.010;
#             #     .000   .940 -0.010;
#             #     .000   .000   .940;
#             # ])
#             sym([ # feedback matrix S
#                 .940 -0.200 -0.000;
#                 .000   .940 -0.199;
#                 .000   .000   .940;
#             ])
#         ),
#         n_seeds,
#         max_time,
#     )

#     # AB
#     results_ab = run_measure(measure_ab)
#     mean_ab = map(r -> r.mean, results_ab)
#     lo_ab = map(r -> r.mean - 0.67*r.std, results_ab)
#     hi_ab = map(r -> r.mean + 0.67*r.std, results_ab)
#     plt_ab = plot(mean_ab', ribbon=[lo_ab hi_ab], linecolor=[:blue :green], labels=['A' 'B'], ylim=(0,1))

#     # ACB
#     results_acb = run_measure(measure_acb)
#     mean_acb = map(r -> r.mean, results_acb)
#     lo_acb = map(r -> r.mean - 0.67*r.std, results_ab)
#     hi_acb = map(r -> r.mean + 0.67*r.std, results_ab)
#     plt_acb = plot(mean_acb', ribbon=[lo_acb hi_acb], linecolor=[:blue :red :green], labels=['A' 'C' 'B'], ylim=(0,1))

#     # plot
#     plot(plt_ab, plt_acb, layout=(1,2))
# end