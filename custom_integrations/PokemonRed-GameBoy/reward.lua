--main functions
function progress()
	frame_count = frame_count + 1

	if (data.audioBank == BATTLE_MUSIC) then
		final_reward = battle_progress()
	else
		final_reward = overworld_progress()
	end

	return final_reward
end

function done_check()

	if data.badges >= 1 then
		return true
	end

	return false
end

function overworld_progress()
	final_reward = moneyReward()
	final_reward = final_reward + partyReward()
	final_reward = final_reward + pkmn1XPReward()
	final_reward = final_reward + overworldMovementReward()
	final_reward = final_reward + explorationReward()
	final_reward = final_reward + timePunishment()
	final_reward = final_reward + badgeReward()
	final_reward = final_reward + OaksParcelReward()

	return final_reward
end

function battle_progress()
	final_reward = hpRatio()
	final_reward = final_reward + levelDifferential()
	final_reward = final_reward + enemyStatusEffect()
	final_reward = final_reward + playerStatusEffect()
	final_reward = final_reward + enemyStatModifiers()
	final_reward = final_reward + playerStatModifiers()

	return final_reward
end

--reward functions
frame_count = 0
MAX_RUN_FRAMES = 15000 -- ~5 minutes?

previous_money = 3000
previous_party_size = 0
previous_pkmn1_exp = 0

previous_xPos = 0
previous_yPos = 0
movement_counter = 0
movement_counter_limit = 120

min_movement_delta = 3

visitedMaps = {}

BATTLE_MUSIC = 8

previous_battle_reward = 0
previous_overworld_reward = 0

has_oaks_parcel = false
current_badges = 0

previous_enemy_status = 0
previous_player_status = 0

previous_enemy_ATK_mod = 7
previous_enemy_DEF_mod = 7
previous_enemy_SPD_mod = 7
previous_enemy_SPEC_mod = 7
previous_enemy_ACC_mod = 7
previous_enemy_EVA_mod = 7

previous_player_ATK_mod = 7
previous_player_DEF_mod = 7
previous_player_SPD_mod = 7
previous_player_SPEC_mod = 7
previous_player_ACC_mod = 7
previous_player_EVA_mod = 7


function moneyReward()
	money_reward = (data.money - previous_money) * 0.0005

	previous_money = data.money
	return money_reward
end

function partyReward()
	party_reward = (data.party_size - previous_party_size) * 1000

	previous_party_size = data.party_size
	return party_reward
end

function pkmn1XPReward()
	xp_reward = (data.totalExpPkmn1 - previous_pkmn1_exp) * 20
	
	previous_pkmn1_exp = data.totalExpPkmn1
	return xp_reward
end

function overworldMovementReward()
	if (movementDeltaOutsideRange(data.xPosOverworld, previous_xPos) or movementDeltaOutsideRange(data.yPosOverworld, previous_yPos)) then
		previous_xPos = data.xPosOverworld
		previous_yPos = data.yPosOverworld
		movement_counter = 0
		final_reward = 0.01
	elseif movement_counter > movement_counter_limit then
		final_reward = -15 - ((movement_counter - movement_counter_limit) * 0.02)
		movement_counter = movement_counter + 1
		
	else
		movement_counter = movement_counter + 1
	end

	return final_reward
end

function movementDeltaOutsideRange(current_pos, last_pos)
	pos_delta = math.abs(current_pos - last_pos)

	return pos_delta >= min_movement_delta
end

-- function overworldMovementReward()
-- 	final_reward = 0
-- 	if (data.xPosOverworld ~= previous_xPos or data.yPosOverworld ~= previous_yPos) then
-- 		previous_xPos = data.xPosOverworld
-- 		previous_yPos = data.yPosOverworld
-- 		movement_counter = 0
-- 	elseif movement_counter > movement_counter_limit then
-- 		movement_counter = movement_counter + 1
-- 		final_reward = -12
-- 	else
-- 		movement_counter = movement_counter + 1
-- 	end

-- 	return final_reward
-- end

function explorationReward()
	final_reward = 0
	if setContains(visitedMaps, data.mapID) then
		addToSet(visitedMaps, data.mapID)
		final_reward = 1500
	end

	return final_reward
end

function timePunishment()
	return -0.05
end

function badgeReward()
	resulting_reward = 0
	if(current_badges ~= data.badges) then
		current_badges = data.badges
		resulting_reward = 2000 * current_badges
	end

	return resulting_reward
end

function OaksParcelReward()
	resulting_reward = 0
	if (not has_oaks_parcel and data.hasOaksParcel > 0) then
		resulting_reward = 1000
		has_oaks_parcel = true
	end

	return resulting_reward
end

function hpRatio()
	-- shouldn't happen but avoids a divide by 0
	if (data.maxHP == 0 or data.enemyMaxHP == 0) then
		return 0
	end


	player_hp_percentage = data.currHP/data.maxHP
	enemy_hp_percentage = data.enemyCurrHP/data.enemyMaxHP
	difference = player_hp_percentage - enemy_hp_percentage

	return difference * 10
end

function levelDifferential()
	difference = data.playerActivePokemonLevel - data.enemyActivePokemonLevel

	return difference * 25
end

function enemyStatusEffect()
	resulting_reward = 0

	if (data.enemyPkmnStatus ~= previous_enemy_status) then
		previous_enemy_status = data.enemyPkmnStatus

		if (previous_enemy_status == 64) then
			resulting_reward = 20 -- paralysis
		elseif (previous_enemy_status == 32) then
			resulting_reward = 90 -- freeze
		elseif (previous_enemy_status == 16) then
			resulting_reward = 40 -- burn
		elseif (previous_enemy_status == 8) then
			resulting_reward = 30 -- poison
		else
			resulting_reward = 70 -- sleep
		end
	end

	return resulting_reward
end

function playerStatusEffect()
	resulting_reward = 0

	if (data.playerPkmnStatus ~= previous_player_status) then
		previous_player_status = data.playerPkmnStatus

		if (previous_player_status == 64) then
			resulting_reward = -20 -- paralysis
		elseif (previous_player_status == 32) then
			resulting_reward = -90 -- freeze
		elseif (previous_player_status == 16) then
			resulting_reward = -40 -- burn
		elseif (previous_player_status == 8) then
			resulting_reward = -30 -- poison
		else
			resulting_reward = -70 --sleep
		end
	end

	return resulting_reward
end

function enemyStatModifiers()
	resulting_reward = 0

	if (data.enemyPkmnATKmod ~= previous_enemy_ATK_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnATKmod - previous_enemy_ATK_mod) * -5)
	end

	if (data.enemyPkmnDEFmod ~= previous_enemy_DEF_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnDEFmod - previous_enemy_DEF_mod) * -5)
	end

	if (data.enemyPkmnSPDmod ~= previous_enemy_SPD_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnSPDmod - previous_enemy_SPD_mod) * -3)
	end

	if (data.enemyPkmnSPECmod ~= previous_enemy_SPEC_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnSPECmod - previous_enemy_SPEC_mod) * -5)
	end

	if (data.enemyPkmnACCmod ~= previous_enemy_ACC_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnACCmod - previous_enemy_ACC_mod) * -5)
	end

	if (data.enemyPkmnEVAmod ~= previous_enemy_EVA_mod) then
		resulting_reward = resulting_reward + ((data.enemyPkmnEVAmod - previous_enemy_EVA_mod) * -2)
	end

	return resulting_reward
end

function playerStatModifiers()
	resulting_reward = 0

	if (data.playerPkmnATKmod ~= previous_player_ATK_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnATKmod - previous_player_ATK_mod) * 10)
	end

	if (data.playerPkmnDEFmod ~= previous_player_DEF_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnDEFmod - previous_player_DEF_mod) * 10)
	end

	if (data.playerPkmnSPDmod ~= previous_player_SPD_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnSPDmod - previous_player_SPD_mod) * 6)
	end

	if (data.playerPkmnSPECmod ~= previous_player_SPEC_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnSPECmod - previous_player_SPEC_mod) * 10)
	end

	if (data.playerPkmnACCmod ~= previous_player_ACC_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnACCmod - previous_player_ACC_mod) * 10)
	end

	if (data.playerPkmnEVAmod ~= previous_player_EVA_mod) then
		resulting_reward = resulting_reward + ((data.playerPkmnEVAmod - previous_player_EVA_mod) * 4)
	end

	return resulting_reward
end

--list helper functions

function addToSet(set, key)
    set[key] = true
end

function removeFromSet(set, key)
    set[key] = nil
end

function setContains(set, key)
    return set[key] ~= nil
end
