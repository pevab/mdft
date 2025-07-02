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
    probabilities :: Vector{Float64}
end

function next(process :: BernoulliAttentionProcess) :: Vector{Float64}
    probabilities = process.probabilities
    n_attributes = length(probabilities)
    result = zeros(n_attributes)
    for a in 1:n_attributes
        p = probabilities[a]
        result[a] = rand(Bernoulli(p))
    end
    # n = count(==(1), result)
    # if n > 0
    #     return result / count(==(1), result)
    # else
    #     return result
    # end
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

function similarity_effect()
    model = MDFT(
        3, # alternatives (cars) A S B
        2, # attributes E (economy) and Q (quality)
        [ # personal evaluation matrix M
            # E     Q
            1.00  3.00; # A
            1.25  2.8; # S
            3.00  1.00; # B
        ],
        # attention weight process
        BernoulliAttentionProcess([.45, .45]),
        [ # contrast matrix C
            1 -0.5 -0.5;
            -0.5 1 -0.5;
            -0.5 -0.5 1;
        ],
        # residual error law
        Dirac(0),
        [ # feedback matrix S
            .940 .000 .000;
            .000 .940 .000;
            .000 .000 .940;
        ]
    )
    function winner(p :: Vector{Float64}) :: Vector{Float64}
        probs = softmax(p)
        w = rand(Categorical(probs))
        r = zeros(3)
        r[w] = 1
        r
    end

    max_time = 100
    n_seeds = 500
    results = zeros((3, max_time))
    for deadline in 1:max_time
        s = zeros((3,0))
        for _ in 1:n_seeds
            history = run(model, DeadlineRule(deadline))
            p = history[:, end]
            s = hcat(s, winner(p))
        end
        results[:, deadline] = mean(s, dims=2)
    end
    plot(results', labels=['A' 'S' 'B'])
end
