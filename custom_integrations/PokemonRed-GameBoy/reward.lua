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
	-- finish once oak's parcel is obtained
	-- if data.hasOaksParcel > 0 then
	if data.party_size >= 1 then
		return true
	-- elseif frame_count > MAX_RUN_FRAMES then
	-- 	return true
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

	return final_reward
end

function battle_progress()
	final_reward = hpRatio()
	final_reward = final_reward + levelDifferential()

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
		print("Exploring a new map!")
		addToSet(visitedMaps, data.mapID)
		final_reward = 1500
	end

	return final_reward
end

function timePunishment()
	return -0.05
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
