using DrWatson
@quickactivate "mdft"

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
    attention_process = BernoulliAttentionProcess([.49, .51])
    model2 = MDFT(
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
    )
    model3 = MDFT(
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
            .940 .000 .000;
            .000 .940 .000;
            .000 .000 .940;
        ])
    )
    function winner(p :: Vector{Float64}) :: Vector{Float64}
        probs = softmax(p)
        w = rand(Categorical(probs))
        r = zeros(size(p))
        r[w] = 1
        r
    end

    max_time = 100
    n_seeds = 1000

    # model2
    results = zeros((2, max_time))
    for deadline in 1:max_time
        s = zeros((2,0))
        for _ in 1:n_seeds
            history = run(model2, DeadlineRule(deadline))
            p = history[:, end]
            s = hcat(s, winner(p))
        end
        results[:, deadline] = mean(s, dims=2)
    end
    plt2 = plot(results', labels=['A' 'B'], linecolor=[:blue :green])

    # model3
    results = zeros((3, max_time))
    for deadline in 1:max_time
        s = zeros((3,0))
        for _ in 1:n_seeds
            history = run(model3, DeadlineRule(deadline))
            p = history[:, end]
            s = hcat(s, winner(p))
        end
        results[:, deadline] = mean(s, dims=2)
    end
    plt3 = plot(results', labels=['A' 'S' 'B'], linecolor=[:blue :red :green])

    # plot
    plot(plt2, plt3, layout=(1,2))
end
