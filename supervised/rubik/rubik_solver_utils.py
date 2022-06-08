from supervised.rubik import gen_rubik_data
from supervised.rubik.gen_rubik_data import make_env_Rubik


def cube_to_string(cube):
    return gen_rubik_data.BOS_LEXEME + gen_rubik_data.cube_bin_to_str(cube) + gen_rubik_data.EOS_LEXEME


def make_RubikEnv():
    return make_env_Rubik(step_limit=1e10, shuffles=100, obs_type='basic')


def generate_problems_rubik(n_problems):
    # 3. Vytváří instanci make_RubikEnv() a "problémy"
    problems = []
    env = make_RubikEnv()
    # loop přes všechny problémy, které přijdou do metody jako parametr
    for _ in range(n_problems):
        # Nejdříve restartuje prostředí (obs = state?)
        obs = env.reset()
        # Prázdné pole epizod
        episode = []
        # vnitřní loop který ukládá mezivýpočty do obs a "_".
        for _ in range(1):
            # Poli epizod se přidávají nějaké data a provede se krok v prostředí, výstup se ukládá do obs
            # Proměnné velkými písmeny jsou nějaké konstanty znaků jako třeba zavináč. Asi jde o dodržení nějakého formátu?
            # gen_rubik_data.cube_bin_to_str(obs) generuje na základě obs rubikovou kostku a dekóduje do stringu.
            episode.append(gen_rubik_data.BOS_LEXEME + gen_rubik_data.cube_bin_to_str(obs) + gen_rubik_data.EOS_LEXEME)
            obs, _, _, _ = env.step(env.action_space.sample()) # Asi zbytečné
        # Do pole problems se přidá epizoda
        problems.append(episode)
    # Metoda vrací problems
    return problems


FACE_TOKENS, MOVE_TOKENS, COL_TO_ID, MOVE_TOKEN_TO_ID = gen_rubik_data.policy_encoding()


def decode_action(raw_action):
    if len(raw_action) < 3:
        # print('Generated invalid move:', raw_action)
        return None

    move = raw_action[2]

    if move not in MOVE_TOKEN_TO_ID:
        # print('Generated invalid move:', raw_action)
        return None

    return MOVE_TOKEN_TO_ID[move]
