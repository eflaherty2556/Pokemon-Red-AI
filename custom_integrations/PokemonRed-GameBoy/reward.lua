movement_counter_limit = 300

previous_money = 3000
previous_party_size = 0
previous_pkmn1_exp = 0

previous_xPos = 0
previous_yPos = 0
movement_counter = 0

function progress()
	final_reward = (data.money - previous_money) * 0.25
	final_reward = final_reward + (data.party_size - previous_party_size) * 100
	final_reward = final_reward + (data.totalExpPkmn1 - previous_pkmn1_exp) * 1.0
	final_reward = overworldMovementReward(final_reward)
	
	return final_reward
end

function done_check()
	-- finish once 6 pokemon are obtained
	if data.party_size == 6 then
		return true
	end
	return false
end

-- helper functions
function overworldMovementReward(final_reward)
	if (data.xPosOverworld ~= previous_xPos or data.yPosOverworld ~= previous_yPos) then
		previous_xPos = data.xPosOverworld
		previous_yPos = data.yPosOverworld
		movement_counter = 0
	elseif movement_counter > movement_counter_limit then
		final_reward = final_reward - 10
		movement_counter = movement_counter + 1
	else
		movement_counter = movement_counter + 1
	end

	return final_reward
end