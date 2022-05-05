<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps({
  title: {
    default: "Definition Title",
  },
  description: {
    default: "Definition Title",
  },
})

const title = ref(props.title)
const description = ref(props.description)
</script>

<template>
  <div class="relative p-4  border-x-2 bg-neutral-100 border-green-500 rounded-lg dark:bg-neutral-800">
    <cite class="flex items-center">
      <div class="flex flex-col items-start">
        <span class="mb-1 text-xl  font-bold">{{ title }}</span>
      </div>
    </cite>
    <p class="mb-4 text-lg">
      {{ description }}
    </p>
  </div>

</template>
