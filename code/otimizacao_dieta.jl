
" ------- Cabeçalho ----------

Alunos: João Pedro Martinez - 1810300; Matheus Nogueira - 1810764
"

" Comentar essa linha ao corrigir "
import Pkg
# Pkg.activate("Otimizacao")

using LinearAlgebra
using SparseArrays, Plots
using JuMP, GLPK, DataFrames
import MathOptInterface as MOI
using NLPModelsJuMP, Random
using CSV
# using Ipopt

Random.seed!(0)

" Questão 1 - Finalizar algoritmo de pontos interiores
"

"Implementa o algoritmo de pontos interiores assumindo problema na forma padrão"
function PontosInteriores(A, b, c; ρ = 0.1, α = 0.95, ϵ = 1e-6, maxIter = 1000, show_iter = true)
    # A: matriz de coeficientes na forma padrão
    # b: vetor de termos independentes na forma padrão
    # c: vetor de custos na forma padrão
    # ρ: parâmetro de redução de gap
    # ϵ: tolerância
    # maxIter: número máximo de iterações
    # show_iter: booleano para mostrar, a cada iteração, o valor dos vetores de solução primal e dual, o valor da função objetivo e o valor do gap
    
    # Dimensões
    m, n = size(A)
    A_ = sparse(A)

    # Inicialização
    dim_x = 1:n
    dim_y = n+1:n+m
    dim_s = n+m+1:2*n+m

    x = ones(n)
    y = ones(m)
    s = ones(n)

    iter = 1
    gap = x' * s
    μ = gap / n
    f_obj = c'*x
    # Vetores de armazenamento
    zk = [[x; y; s]]
    Fk = []
    
    converge = 0

    # Iteração
    while converge == 0
        
        if show_iter
            println("Iteração $iter")
            println("   Vetor solução primal = $x")
            println("   Vetor solução dual = $y")
            println("   GAP = $gap")
            println("   Função objetivo = ",c'x)
        end

        F = [A_*x - b ; A_'*y + s - c ; diagm(x)*diagm(s)*ones(n) - μ*ones(n)]
        push!(Fk, F)
        # Teste de convergência
        if maximum(abs.(F[1:n+m])) < ϵ && gap < ϵ
            converge = 1
            f_obj = c'*x
            println("Optimal solution found!")
            break
        end
        
        # Teste de número máximo de iterações
        if iter >= maxIter
            converge = 2
            f_obj = c'*x
            println("Reached max number of iterations!")
            break
        end

        # Teste de ilimitado
        if maximum(abs.(x)) >= 1e15
            converge = 3
            f_obj = missing
            println("Problem is unlimited")
            break
        end
    
        # Teste de inviavel
        if maximum(abs.(y)) >= 1e15
            converge = 4
            f_obj = missing
            println("Problem is unfeasible")
            break
        end
        # Passo de Newton
        J = spzeros(2*n+m, 2*n+m)
        J[1:m, 1:n] = A_
        J[m+1:m+n, n+1:n+m] = A_'
        J[m+1:m+n, n+m+1:m+2*n] = sparse(I, n, n)
        J[n+m+1:m+2*n, 1:n] = sparse(diagm(s))
        J[n+m+1:m+2*n, n+m+1:m+2*n] = sparse(diagm(x))

        # Resolução do sistema linear
        d = J\-F
        
        # Atualização
        dx = d[dim_x]
        dy = d[dim_y]
        ds = d[dim_s]
        B_p = minimum([1 ; α.*(-x[dx .< 0] ./ dx[dx .< 0])])
        B_d = minimum([1 ; α.*(-s[ds .< 0] ./ ds[ds .< 0])])

        x += B_p*dx
        y += B_d*dy
        s += B_d*ds

        push!(zk, [x; y; s])
        gap = x' * s
        μ = ρ*gap / n

        iter += 1
    end
    return x, y, s, iter, gap, converge, zk, Fk, f_obj
end

" Funcao para, a partir de um modelo definido na forma canonica, obter as matrizes e vetores na forma padrão necessária para os pontos interiores" 
function get_std_form(m)
    
    # Extração da Forma Padrão
    nlp = MathOptNLPModel(m) # the input "< model >" is the name of the model you created by JuMP before with variables and constraints (and optionally the objective function) attached to it.
    x = zeros(nlp.meta.nvar)
    c = NLPModelsJuMP.grad(nlp, x)
    A = Matrix(NLPModelsJuMP.jac(nlp, x))
    
    lb = nlp.meta.lcon
    ub = nlp.meta.ucon
    m,n = size(A)
    b = deepcopy(lb)
    for i = 1:m
        v_aux = zeros(m)
        l = lb[i]
        u = ub[i]
        if l==u && !isinf(l)
            continue
        end
        if isinf(l)
            b[i] = ub[i]
            v_aux[i] = 1
            A = hcat(A, v_aux)
        end
        if isinf(u)
            v_aux[i] = -1
            A = hcat(A, v_aux)
        end
        # display(A)
    end
    c = vcat(c, zeros(size(A)[2]-n))

    return A, b, c
end


" Questão 3:

Selecione um problema de interesse, descreva e apresente a formulação, apresente os dados de entrada, e apresente sua solução. Mostre como utilizar a
informação dual no seu caso e comente os resultados. Monte uma apresentação de 15 min para apresentar esses resultados. Este item poderá ser realizado em dupla, mas
somente mediante a entrega individual dos itens 1 e 2. O critério de avaliação deste item será: 40% sobre a compreensão do problema (da sua formulação, dos dados
apresentados, e dos resultados), 20% sobre a qualidade da apresentação (formato, organização, etc.) e 40% para a demonstração de que o seu algoritmo obteve os
resultados corretos, comparando com o Gurobi (ou outro solver open source ), tempo computacional, etc.

"

function harris_benedict_equation(altura, peso, idade, sexo)
    if sexo == "Homem"
        return 66.4730 + (13.7516 * peso) + (5.0033 * altura) - (6.7550 * idade)
    else
        return 655.0955 + (9.5634 * peso) + (1.8496 * altura) - (4.6756 * idade)
    end
end

function kcal_by_activity_level(bmr,dias_atividade)
    if dias_atividade == 0
        mult = 1.2
    elseif (dias_atividade > 0) & (dias_atividade <=3)
        mult = 1.375
    elseif (dias_atividade > 3) & (dias_atividade <= 5)
        mult = 1.55
    elseif (dias_atividade > 5) & (dias_atividade <=7)
        mult = 1.725
    else
        mult = 1.9
    end
    return mult*bmr
end

function kcal_diaria_bms(altura, peso, idade, sexo, dias_atividade)
    bmr = harris_benedict_equation(altura, peso, idade, sexo)
    kcal = kcal_by_activity_level(bmr, dias_atividade)
    return kcal
end

function get_daily_intakes_by_food(x, n)
    quantidades_diarias = JuMP.value.(x)
    df_quantidades_diarias_otimas = DataFrame(zeros(n,7),:auto)
    rename!(df_quantidades_diarias_otimas, ["Dia $i"  for i in 1:7])
    lista_quantidades_diarias = [quantidades_diarias[i:i+6]  for i in 1:7:n*7]
    for i in 1:n
        df_quantidades_diarias_otimas[i,:] = lista_quantidades_diarias[i]
    end
    insertcols!(df_quantidades_diarias_otimas, 1, :Nome => dados[:,"Nome"])
    for i in 1:7
        df_quantidades_diarias_otimas[:,"Dia $i"] = [val <= 1e-4 ? 0 : val for val in df_quantidades_diarias_otimas[:,"Dia $i"]]
    end
    return df_quantidades_diarias_otimas
end

function get_total_daily_intake_g(optimal_daily_intakes)
    return describe(optimal_daily_intakes, sum=>:sum)
end

function get_macros_daily_intakes(dados, optimal_daily_intakes)
    df_macros_daily_intakes = DataFrame(zeros(3, 7), :auto);
    rename!(df_macros_daily_intakes, ["Dia $i" for i in 1:7]);
    for i in 1:7
        df_macros_daily_intakes[1, "Dia $i"] = dados[:,"Proteina por Grama"]'*optimal_daily_intakes[:,"Dia $i"];
        df_macros_daily_intakes[2, "Dia $i"] = dados[:,"Carboidrato por Grama"]'*optimal_daily_intakes[:,"Dia $i"];
        df_macros_daily_intakes[3, "Dia $i"] = dados[:,"Gordura por Grama"]'*optimal_daily_intakes[:,"Dia $i"];
    end
    insertcols!(df_macros_daily_intakes, 1, :Macros => ["Prot"; "Carb"; "Gord"])
    return df_macros_daily_intakes
end

function get_macros_g_kg_ratio(df_macros_daily_intakes, peso)
    df_macros_g_kg_ratio = df_macros_daily_intakes[!, 2:end]./peso;
    insertcols!(df_macros_g_kg_ratio, 1, :Macros => ["Prot"; "Carb"; "Gord"])
    return df_macros_g_kg_ratio
end

function get_macros_daily_kcal_perc(df_macros_daily_intakes)
    df_tot_kcal_per_macro = df_macros_daily_intakes[!, 2:end].*[4; 4; 9]
    tot_kcal_per_day = describe(df_tot_kcal_per_macro, sum=>:sum)[:,"sum"]
    for i in 1:7
        df_tot_kcal_per_macro[:, "Dia $i"] = df_tot_kcal_per_macro[:, "Dia $i"]./tot_kcal_per_day[i]
    end 
    return df_tot_kcal_per_macro
end

function run_GLPK_solver(dados, kcal_diaria, REF_CARB_KG, REF_PROT_KG, REF_GORD_KG, c_obj, n)
    SUM_CARB_DAILY = REF_CARB_KG * peso
    SUM_PROT_DAILY = REF_PROT_KG * peso
    SUM_GORD_DAILY = REF_GORD_KG * peso

    LB_WEEK = dados[:,"LB Semanal"]
    UB_WEEK = dados[:,"UB Semanal"]
    LB_DAILY_1 = dados[:,"LB Dia 1"]
    UB_DAILY_1 = dados[:,"UB Dia 1"]
    LB_DAILY_2 = dados[:,"LB Dia 2"]
    UB_DAILY_2 = dados[:,"UB Dia 2"]

    days_1 = [1,3,5,7]
    
    model = Model(() -> GLPK.Optimizer(method=GLPK.INTERIOR))
    @variable(model, x[i=1:n*7]>=0)
    @objective(model, Min, c_obj'*x)
 
    for p in 1:n
        # Restricao de LB e UB semanal por produto    
        @constraint(model,LB_WEEK[p] <= sum([x[i] for i in 7*(p-1)+1:7*p]))  
        @constraint(model,UB_WEEK[p] >= sum([x[i] for i in 7*(p-1)+1:7*p]))   
    
        for i in 1:7
            if i in days_1
                @constraint(model, x[(p-1)*7+i] <= UB_DAILY_1[p])  
            else
                @constraint(model, x[(p-1)*7+i] <= UB_DAILY_2[p]) 
            end
        end
    end
    
    for i in 1:7
        s_prot = dados[:,"Proteina por Grama"]'*x[collect(i:7:7*n)]
        s_gord = dados[:,"Gordura por Grama"]'*x[collect(i:7:7*n)]
        s_carb = dados[:,"Carboidrato por Grama"]'*x[collect(i:7:7*n)]
    
        # Restricoes para cada macronutriente
        @constraint(model, s_carb <= 1.1*SUM_CARB_DAILY)
        @constraint(model, s_carb >= 0.9*SUM_CARB_DAILY)
    
        @constraint(model, s_prot <= 1.1*SUM_PROT_DAILY)
        @constraint(model, s_prot >= 0.9*SUM_PROT_DAILY)
    
        @constraint(model, s_gord <= 1.1*SUM_GORD_DAILY)
        @constraint(model, s_gord >= 0.9*SUM_GORD_DAILY)
    
        #Restricao de caloria diaria
        @constraint(model, 4*s_carb + 4*s_prot + 9*s_gord <= 1.1*kcal_diaria)
        @constraint(model, 4*s_carb + 4*s_prot + 9*s_gord >= 0.9*kcal_diaria)
    end
    
    tempo_glpk = @elapsed optimize!(model)
    custo_dieta_glpk = c_obj'*JuMP.value.(x)
    optimal_daily_intakes = get_daily_intakes_by_food(x, n)

    return model, tempo_glpk, custo_dieta_glpk, optimal_daily_intakes
end


"Código utilizado apenas na primeira versão do problema, sem diferenciação dos dias"
" Ele foi deixado aqui apenas porque o utilizamos na apresentação"

# " Versão 1 - sem diferenciar dia ativo e descanso"

# # path_dir = "C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\11° Periodo\\Otimizacao\\Lista3\\"
# # path_dir = "D:\\matheuscn\\OneDrive\\Documentos\\PUC\\0_Períodos\\11° Periodo\\Otimizacao\\Lista3\\"
# # dados = CSV.read(path_dir*"precos_alimentos_v1.csv",DataFrame)
# dados = CSV.read("precos_alimentos_v1.csv",DataFrame)

# peso = 80
# altura = 165
# idade = 23
# atividade_fisica = 4
# sexo = "Homem"
# kcal_diaria = kcal_diaria_bms(altura, peso, idade, sexo, atividade_fisica)

# REF_PROT_KG = 2
# REF_GORD_KG = 1

# SUM_PROT_DAILY = REF_PROT_KG * peso
# SUM_GORD_DAILY = REF_GORD_KG * peso

# dados[:,"Proteina por Grama"] = dados[:,"Proteina(g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]
# dados[:,"Carboidrato por Grama"] = dados[:,"Carboidrato (g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]
# dados[:,"Gordura por Grama"] = dados[:,"Gordura (g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]

# LB_WEEK = dados[:,"LB Semanal"]
# UB_WEEK = dados[:,"Upper Bound Semanal"]
# LB_DAILY = dados[:,"Lower Bound Diario"]
# UB_DAILY = dados[:,"Upper Bound Diario"]

# c = dados[:,"Preco Quantidade Mercado (Reais)"]./dados[:,"Quantidade Unidade Mercado (g)"]
# n = size(dados,1)

# c_obj = vcat([ [c[idx] for i = 1:7] for idx = 1:length(c) ]...)


# model = Model(() -> GLPK.Optimizer(method=GLPK.INTERIOR))
# @variable(model, x[i=1:n*7]>=0)
# @objective(model, Min, c_obj'*x)
# for p in 1:n
#     # Restricao de LB e UB semanal por produto    
#     @constraint(model,LB_WEEK[p] <= sum([x[i] for i in 7*(p-1)+1:7*p]))  
#     @constraint(model,UB_WEEK[p] >= sum([x[i] for i in 7*(p-1)+1:7*p]))   

#     for i in 1:7
#         # Restricao de LB e UB diaria por produto
#         @constraint(model, x[(p-1)*7+i] <= UB_DAILY[p])  
#         # @constraint(model, LB_DAILY[p] <= x[(p-1)*7+i])    
#     end
# end

# for i in 1:7
#     s_prot = dados[:,"Proteina por Grama"]'*x[collect(i:7:7*n)]
#     s_gord = dados[:,"Gordura por Grama"]'*x[collect(i:7:7*n)]
#     s_carb = dados[:,"Carboidrato por Grama"]'*x[collect(i:7:7*n)]

#     # Restricoes para cada macronutriente

#     @constraint(model, s_prot == SUM_PROT_DAILY)
#     # @constraint(model, s_prot <= 1.01*SUM_PROT_DAILY)
#     # @constraint(model, s_prot >= 0.99*SUM_PROT_DAILY)

#     @constraint(model, s_gord == SUM_GORD_DAILY)
#     # @constraint(model, s_gord <= 1.01*SUM_GORD_DAILY)
#     # @constraint(model, s_gord >= 0.99*SUM_GORD_DAILY)

#     #Restricao de caloria diaria
#     @constraint(model, 4*s_carb + 4*s_prot + 9*s_gord == kcal_diaria)
# end

# model

# tempo_glpk = @elapsed optimize!(model)

# custo_dieta_glpk = c_obj'*JuMP.value.(x)

# quantidades_diarias = JuMP.value.(x)

# df_quantidades_diarias_otimas = DataFrame(zeros(n,7),:auto)
# rename!(df_quantidades_diarias_otimas, ["Dia $i"  for i in 1:7])
# lista_quantidades_diarias = [quantidades_diarias[i:i+6]  for i in 1:7:n*7]
# for i in 1:n
#     df_quantidades_diarias_otimas[i,:] = lista_quantidades_diarias[i]
# end

# df_quantidades_diarias_otimas[:,"Nome"] = dados[:,"Nome"]
# CSV.write("qtd_diarias_v1.csv",df_quantidades_diarias_otimas)





" Versão 3 - dias ativos e de descanso variando Scarb"


# path_dir = "C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\11° Periodo\\Otimizacao\\Lista3\\"
# path_dir = "D:\\matheuscn\\OneDrive\\Documentos\\PUC\\0_Períodos\\11° Periodo\\Otimizacao\\Lista3\\"
dados = CSV.read("precos_alimentos_ativo_descanso_v3.csv",DataFrame)


# Print the types
for (column_name, column_type) in zip(names(dados), eltype.(eachcol(dados)))
    println("$column_name: $column_type")
end

dados[:,"Proteina por Grama"] = dados[:,"Proteina(g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]
dados[:,"Carboidrato por Grama"] = dados[:,"Carboidrato (g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]
dados[:,"Gordura por Grama"] = dados[:,"Gordura (g)"] ./ dados[:,"Quantidade Tabela Nutricional (g)"]

c = dados[:,"Preco Quantidade Mercado (Reais)"]./dados[:,"Quantidade Unidade Mercado (g)"]
c_obj = vcat([ [c[idx] for i = 1:7] for idx = 1:length(c) ]...)

# Numero de produtos considerados na base de dados
n = size(dados,1)

# ======================= Otimização de Dieta - Indivíduo 1 ====================================

# >>> Solução do problema via GLPK
peso = 80
altura = 165
idade = 23
atividade_fisica = 4
sexo = "Homem"
kcal_diaria = kcal_diaria_bms(altura, peso, idade, sexo, atividade_fisica)
REF_CARB_KG = 4
REF_PROT_KG = 2
REF_GORD_KG = 1

model, tempo_glpk, custo_dieta_glpk, optimal_daily_intakes = run_GLPK_solver(dados, kcal_diaria, REF_CARB_KG, REF_PROT_KG, REF_GORD_KG, c_obj, n)

# Dataframe com a dieta recomendada
CSV.write("dieta_diaria_individuo_1.csv", optimal_daily_intakes)
# Dataframe com o total de comida consumida em cada dia (gramas)
df_total_daily_intake_g = get_total_daily_intake_g(optimal_daily_intakes)
df_total_daily_intake_g[1,2] = 0
CSV.write("total_gramas_diario_individuo_1.csv", df_total_daily_intake_g)
# Dataframe com o cosumo diário, em gramas, de cada macronutriente
df_macros_daily_intakes = get_macros_daily_intakes(dados, optimal_daily_intakes)
CSV.write("total_gramas_macro_diario_individuo_1.csv", df_macros_daily_intakes)
# Dataframe com a proporcao diária g/kg de cada macronutriente
df_macros_g_kg_ratio = get_macros_g_kg_ratio(df_macros_daily_intakes, peso)
CSV.write("total_ratio_gkg_macros_diario_individuo_1.csv", df_macros_g_kg_ratio)
# Dataframe com as porcentagens de kcal diária de cada macronutriente
df_macros_daily_kcal_perc = get_macros_daily_kcal_perc(df_macros_daily_intakes)
CSV.write("total_perc_macros_diario_individuo_1.csv", df_macros_daily_kcal_perc)

# >>> Solução do problema via método dos Pontos Interiores
A,b,c = get_std_form(model);
tempo_PI = @elapsed x, y, s, iter, gap, converge, zk, Fk = PontosInteriores(A, b, c; show_iter = false);
custo_dieta_PI = c_obj'*x[1:n*7]

println("Comparacao entre GLPK e Pontos Interiores (PI)")
println("Δt de execucao (GLPK):", tempo_glpk)
println("Δt de execucao (PI):", tempo_PI)
println("Valor da func. Objetivo (GLPK):", custo_dieta_glpk)
println("Valor da func. Objetivo (PI):", custo_dieta_PI)

df_resumo_individuo_1 = DataFrame(Dict("Metodo"=>["GLPK", "PI"],"Custo Mínimo"=>[custo_dieta_glpk,custo_dieta_PI],
                                        "Tempo"=>[tempo_glpk,tempo_PI]))
CSV.write("resumo_otimizacao_individuo_1.csv",df_resumo_individuo_1)

# ======================= Otimização de Dieta - Indivíduo 2 ====================================

# >>> Solução do problema via GLPK
peso = 110
altura = 190
idade = 44
atividade_fisica = 8
sexo = "Homem"
kcal_diaria = kcal_diaria_bms(altura, peso, idade, sexo, atividade_fisica)
REF_CARB_KG = 4
REF_PROT_KG = 2
REF_GORD_KG = 1

model, tempo_glpk, custo_dieta_glpk, optimal_daily_intakes = run_GLPK_solver(dados, kcal_diaria, REF_CARB_KG, REF_PROT_KG, REF_GORD_KG, c_obj, n)

# Dataframe com a dieta recomendada
CSV.write("dieta_diaria_individuo_2.csv", optimal_daily_intakes)
# Dataframe com o total de comida consumida em cada dia (gramas)
df_total_daily_intake_g = get_total_daily_intake_g(optimal_daily_intakes)
df_total_daily_intake_g[1,2] = 0
CSV.write("total_gramas_diario_individuo_2.csv", df_total_daily_intake_g)
# Dataframe com o cosumo diário, em gramas, de cada macronutriente
df_macros_daily_intakes = get_macros_daily_intakes(dados, optimal_daily_intakes)
CSV.write("total_gramas_macro_diario_individuo_2.csv", df_macros_daily_intakes)
# Dataframe com a proporcao diária g/kg de cada macronutriente
df_macros_g_kg_ratio = get_macros_g_kg_ratio(df_macros_daily_intakes, peso)
CSV.write("total_ratio_gkg_macros_diario_individuo_2.csv", df_macros_g_kg_ratio)
# Dataframe com as porcentagens de kcal diária de cada macronutriente
df_macros_daily_kcal_perc = get_macros_daily_kcal_perc(df_macros_daily_intakes)
CSV.write("total_perc_macros_diario_individuo_2.csv", df_macros_daily_kcal_perc)


# >>> Solução do problema via método dos Pontos Interiores
A,b,c = get_std_form(model);
tempo_PI = @elapsed x, y, s, iter, gap, converge, zk, Fk = PontosInteriores(A, b, c; show_iter = false);
custo_dieta_PI = c_obj'*x[1:n*7]

println("Comparacao entre GLPK e Pontos Interiores (PI)")
println("Δt de execucao (GLPK):", tempo_glpk)
println("Δt de execucao (PI):", tempo_PI)
println("Valor da func. Objetivo (GLPK):", custo_dieta_glpk)
println("Valor da func. Objetivo (PI):", custo_dieta_PI)

df_resumo_individuo_2 = DataFrame(Dict("Metodo"=>["GLPK", "PI"],"Custo Mínimo"=>[custo_dieta_glpk,custo_dieta_PI],
                                        "Tempo"=>[tempo_glpk,tempo_PI]))
CSV.write("resumo_otimizacao_individuo_2.csv",df_resumo_individuo_2)

# ======================= Otimização de Dieta - Indivíduo 3 ====================================

# >>> Solução do problema via GLPK
peso = 85
altura = 166
idade = 32
atividade_fisica = 2
sexo = "Mulher"
kcal_diaria = kcal_diaria_bms(altura, peso, idade, sexo, atividade_fisica)
REF_CARB_KG = 3.35
REF_PROT_KG = 1.8
REF_GORD_KG = 1.1

model, tempo_glpk, custo_dieta_glpk, optimal_daily_intakes = run_GLPK_solver(dados, kcal_diaria, REF_CARB_KG, REF_PROT_KG, REF_GORD_KG, c_obj, n)

# Dataframe com a dieta recomendada
CSV.write("dieta_diaria_individuo_3.csv", optimal_daily_intakes)
# Dataframe com o total de comida consumida em cada dia (gramas)
df_total_daily_intake_g = get_total_daily_intake_g(optimal_daily_intakes)
df_total_daily_intake_g[1,2] = 0
CSV.write("total_gramas_diario_individuo_3.csv", df_total_daily_intake_g)
# Dataframe com o cosumo diário, em gramas, de cada macronutriente
df_macros_daily_intakes = get_macros_daily_intakes(dados, optimal_daily_intakes)
CSV.write("total_gramas_macro_diario_individuo_3.csv", df_macros_daily_intakes)
# Dataframe com a proporcao diária g/kg de cada macronutriente
df_macros_g_kg_ratio = get_macros_g_kg_ratio(df_macros_daily_intakes, peso)
CSV.write("total_ratio_gkg_macros_diario_individuo_3.csv", df_macros_g_kg_ratio)
# Dataframe com as porcentagens de kcal diária de cada macronutriente
df_macros_daily_kcal_perc = get_macros_daily_kcal_perc(df_macros_daily_intakes)
CSV.write("total_perc_macros_diario_individuo_3.csv", df_macros_daily_kcal_perc)


# >>> Solução do problema via método dos Pontos Interiores
A,b,c = get_std_form(model);
tempo_PI = @elapsed x, y, s, iter, gap, converge, zk, Fk = PontosInteriores(A, b, c; show_iter = false);
custo_dieta_PI = c_obj'*x[1:n*7]

println("Comparacao entre GLPK e Pontos Interiores (PI)")
println("Δt de execucao (GLPK):", tempo_glpk)
println("Δt de execucao (PI):", tempo_PI)
println("Valor da func. Objetivo (GLPK):", custo_dieta_glpk)
println("Valor da func. Objetivo (PI):", custo_dieta_PI)

df_resumo_individuo_3 = DataFrame(Dict("Metodo"=>["GLPK", "PI"],"Custo Mínimo"=>[custo_dieta_glpk,custo_dieta_PI],
                                        "Tempo"=>[tempo_glpk,tempo_PI]))
CSV.write("resumo_otimizacao_individuo_3.csv",df_resumo_individuo_3)